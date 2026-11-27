from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def last_valid_hidden(hidden, attention_mask):
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
    last = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    if (last < 0).any():
        raise ValueError("empty_text_sequence")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class RewardModel(nn.Module):
    def __init__(self, backbone, ontology):
        super().__init__(); self.backbone = backbone; self.ontology = ontology
        size = backbone.config.text_config.hidden_size
        dims = {"A": 1, "T": 1, **{k: len(ontology[k]) for k in
                ("failure_actions", "failure_types", "failure_results", "recovery")}}
        self.heads = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(size), nn.Linear(size, n)) for k, n in dims.items()})

    def forward(self, **inputs):
        output = self.backbone(**inputs, use_cache=False, return_dict=True)
        state = last_valid_hidden(output.last_hidden_state, inputs["attention_mask"]).float()
        logits = {k: layer(state) for k, layer in self.heads.items()}
        logits["A"] = logits["A"].squeeze(-1); logits["T"] = logits["T"].squeeze(-1)
        return logits

    def save_weights(self, folder, config):
        from safetensors.torch import save_file
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(folder / "adapter", safe_serialization=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.heads.state_dict().items()}, str(folder / "heads.safetensors"))
        (folder / "reward_model.json").write_text(json.dumps({"config": config, "ontology": self.ontology}, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_weights(self, folder):
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        from safetensors.torch import load_file
        folder = Path(folder)
        metadata = json.loads((folder / "reward_model.json").read_text(encoding="utf-8"))
        if metadata["ontology"]["sha256"] != self.ontology["sha256"]:
            raise ValueError("checkpoint_ontology_mismatch")
        set_peft_model_state_dict(self.backbone, load_peft_weights(str(folder / "adapter")))
        self.heads.load_state_dict(load_file(str(folder / "heads.safetensors")))


def build_model(config, ontology):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLModel
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[config["mixed_precision"]]
    local_only = config.get("local_files_only", False)
    from dual_rm.pretrained import compatible_config
    pretrained_config = compatible_config(config)
    conditional = "Qwen3VLForConditionalGeneration" in (pretrained_config.architectures or [])
    loader = Qwen3VLForConditionalGeneration if conditional else Qwen3VLModel
    loaded, loading_info = loader.from_pretrained(config["model_name_or_path"], config=pretrained_config, dtype=dtype,
        attn_implementation="sdpa", local_files_only=local_only, output_loading_info=True)
    # In Transformers 4.57 Qwen3VLModel has an empty base_model_prefix, so it
    # cannot strip the official Instruct checkpoint's "model." prefix itself.
    # Load the checkpoint's declared architecture before extracting its trunk.
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("pretrained_weights_not_fully_loaded:" + json.dumps(loading_info)[:2000])
    backbone = loaded.model if conditional else loaded
    lora = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                      lora_dropout=0., target_modules=r"language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)")
    backbone = get_peft_model(backbone, lora)
    if config.get("gradient_checkpointing", True):
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not any("lora_" in n and p.requires_grad for n, p in backbone.named_parameters()):
        raise ValueError("no_language_LoRA_parameters_matched")
    if any(p.requires_grad for n, p in backbone.named_parameters() if ".visual." in n):
        raise ValueError("vision_encoder_must_be_frozen")
    reward = RewardModel(backbone, ontology)
    reward.pretrained_loading_info = {"architecture": loader.__name__, **loading_info}
    return reward


def masked_mean(values, mask, density=None):
    weights = mask.to(values.dtype)
    numerator = (values * weights).sum()
    if density is None:
        denominator = weights.sum().detach()
    else:
        # A fixed training-set density yields the same objective with different
        # microbatch / accumulation sizes. Per-microbatch masked means would
        # otherwise underweight rare terminal T labels differently for 4B/8B.
        denominator = values.new_tensor(values.numel() * density if density > 0 else 1.)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(denominator)
        numerator = numerator * torch.distributed.get_world_size()
    return numerator / (denominator.clamp_min(1) if density is None else denominator)


def loss_fn(logits, targets, config):
    if config['method'] == 'native_sft':
        loss = logits['native_loss'].mean()
        return loss, {'native_answer_cross_entropy': loss}
    density = config.get('supervision_density', {})
    parts = {"A": masked_mean(F.mse_loss(logits["A"].float().sigmoid(), targets["A_value"], reduction="none"), targets["A_mask"], density.get('A')),
             "T": masked_mean(F.binary_cross_entropy_with_logits(logits["T"].float(), targets["T_value"], reduction="none"), targets["T_mask"], density.get('T'))}
    for head in ("failure_actions", "failure_types", "failure_results"):
        parts[head] = masked_mean(F.binary_cross_entropy_with_logits(logits[head].float(), targets[head + "_value"], reduction="none"), targets[head + "_mask"], density.get(head))
    parts["recovery"] = masked_mean(F.cross_entropy(logits["recovery"].float(), targets["recovery_value"], reduction="none"), targets["recovery_mask"], density.get('recovery'))
    # Keep every head in the autograd graph so distributed training also works
    # when a rank's current batch has no terminal or auxiliary supervision.
    total = sum(x.sum() * 0 for x in logits.values())
    weights = config["loss_weights"]
    for name, value in parts.items():
        weight = weights.get(name, 0.) if config["method"] == "joint" or name in ("A", "T") else 0.
        total = total + weight * value
    return total, parts
