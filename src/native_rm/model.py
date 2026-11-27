from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from dual_rm.pretrained import load_conditional


class NativeReward(nn.Module):
    def __init__(self, generator, ontology):
        super().__init__()
        self.generator, self.ontology = generator, ontology
        self.heads = nn.ModuleDict()

    def forward(self, labels, **inputs):
        # Project only positions that can contribute to the answer loss.
        # This avoids materializing [batch, all_image_tokens, vocabulary].
        positions = (labels[:, 1:] != -100).any(0).nonzero().flatten()
        if positions.numel() == 0:
            raise ValueError('no_supervised_native_answer_tokens')
        result = self.generator(**inputs, use_cache=False, logits_to_keep=positions, return_dict=True)
        target = labels[:, positions + 1]
        loss = F.cross_entropy(result.logits.float().transpose(1, 2), target, ignore_index=-100, reduction='none')
        count = (target != -100).sum(1)
        if not count.gt(0).all():
            raise ValueError('native_batch_contains_unsupervised_sample')
        nll = loss.sum(1)
        return {'native_loss': nll / count, 'native_nll': nll, 'native_token_count': count}

    def save_weights(self, folder, config):
        from safetensors.torch import save_file
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        self.generator.save_pretrained(folder / 'adapter', safe_serialization=True)
        # Common checkpoint fingerprint API; this is a format marker, not a
        # newly initialized reward head. The native LM output head is retained.
        save_file({'native_sft_format': torch.tensor([1])}, str(folder / 'heads.safetensors'))
        (folder / 'reward_model.json').write_text(json.dumps({'config': config, 'ontology': self.ontology}, indent=2))

    def load_weights(self, folder):
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        set_peft_model_state_dict(self.generator, load_peft_weights(str(Path(folder) / 'adapter')))


def build_model(config, ontology):
    from peft import LoraConfig, TaskType, get_peft_model
    base, info = load_conditional(config)
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=config['lora_rank'],
        lora_alpha=config['lora_alpha'], lora_dropout=0.,
        target_modules=r'model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'))
    if config.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if not any('lora_' in n and p.requires_grad for n,p in model.named_parameters()):
        raise ValueError('native_language_LoRA_not_matched')
    if any(p.requires_grad for n,p in model.named_parameters() if '.visual.' in n or 'lm_head' in n):
        raise ValueError('native_vision_and_LM_output_head_must_remain_frozen')
    wrapped = NativeReward(model, ontology)
    wrapped.pretrained_loading_info = info
    return wrapped


def evaluate(model, loader, accelerator):
    model.eval(); losses, nll, counts = [], [], []
    with torch.inference_mode():
        for batch in loader:
            result = accelerator.gather_for_metrics(model(**batch['inputs']))
            losses.extend(result['native_loss'].float().cpu().tolist())
            nll.extend(result['native_nll'].float().cpu().tolist())
            counts.extend(result['native_token_count'].cpu().tolist())
    model.train()
    return {'split': 'validation', 'native_sft_samples': len(losses),
            'native_sft_loss': sum(losses) / len(losses),
            'native_answer_tokens': sum(counts), 'native_token_nll': sum(nll) / sum(counts),
            'threshold_selection_requires_separate_native_generation_evaluation': True}
