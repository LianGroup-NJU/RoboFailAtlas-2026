from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from qwen_rm.data import FrameCache


def progress_probability(logits):
    """Keep native discrete head; regress its differentiable bin expectation."""
    centers = torch.linspace(0, 1, logits.shape[-1], device=logits.device, dtype=torch.float32)
    return (logits.float().softmax(-1) * centers).sum(-1)


def prompt_messages(row):
    # Preserve the pretrained Robometer prompt and per-image progress tokens.
    prompt = (f"The task for the robot is '{row['instruction']}'. Given the trajectory video, "
              'predict the task progress at each frame, how far along the robot is towards '
              'completing the task, a float between 0 and 1, where 0 is the starting state '
              'and 1 is when the task is completed. If the robot is not performing the '
              'same task, predict 0 progress.')
    content = [{'type': 'text', 'text': prompt}]
    for _ in row['frames']:
        content.extend([{'type': 'image'}, {'type': 'text', 'text': '<|prog_token|>'}])
    return [{'role': 'user', 'content': content}]


class RobometerCollator:
    def __init__(self, processor, frame_cache, max_tokens=8192):
        self.processor, self.cache, self.max_tokens = processor, FrameCache(frame_cache), max_tokens

    def __call__(self, rows):
        from PIL import Image
        from qwen_rm.data import supervised_targets
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            groups[len(row['frames'])].append((index, row))
        inputs = []
        # Native RBM stacks frame logits. Separate variable-length prefixes so
        # a batch containing 1-frame and 8-frame histories does not pad/repeat RGB.
        for count, items in sorted(groups.items()):
            texts, images = [], []
            for _, row in items:
                texts.append(self.processor.apply_chat_template(prompt_messages(row), tokenize=False,
                    add_generation_prompt=False, add_vision_id=True, enable_thinking=False, fps=1))
                for path in self.cache.paths(row):
                    with Image.open(path) as pic:
                        images.append(pic.convert('RGB').copy())
            batch = dict(self.processor(text=texts, images=images, padding=True,
                                        truncation=False, return_tensors='pt'))
            batch.pop('token_type_ids', None)
            if batch['input_ids'].shape[-1] > self.max_tokens:
                raise ValueError('multimodal_sequence_over_budget_do_not_truncate')
            token = self.processor.tokenizer.convert_tokens_to_ids('<|prog_token|>')
            if not torch.all((batch['input_ids'] == token).sum(-1) == count):
                raise ValueError('robometer_progress_tokens_do_not_match_history_frames')
            inputs.append({'indices': torch.tensor([i for i, _ in items]),
                           'frame_count': count, 'batch': batch})
        return {'inputs': {'groups': inputs, 'batch_size': len(rows)}, 'targets': supervised_targets(rows),
                'row_ids': [r['id'] for r in rows], 'case_ids': [r['case_id'] for r in rows],
                'frames': [r['frame'] for r in rows]}


class RobometerAT(nn.Module):
    def __init__(self, rbm, ontology):
        super().__init__()
        self.rbm, self.ontology = rbm, ontology

    @property
    def heads(self):
        return nn.ModuleDict({'progress': self.rbm.progress_head, 'success': self.rbm.success_head})

    def forward(self, groups, batch_size):
        rows = [None] * batch_size
        for group in groups:
            result, _ = self.rbm(**group['batch'], sample_type='progress', use_cache=False)
            a, t = result.progress_logits['A'], result.success_logits['A']
            # Upstream returns [B,bins] when every sample has exactly one
            # progress token, but [B,T,bins] for longer histories.
            if group['frame_count'] == 1:
                if a.ndim == 2:
                    a = a.unsqueeze(1)
                if t.ndim == 1:
                    t = t.unsqueeze(1)
            if a.shape[:2] != (len(group['indices']), group['frame_count']):
                raise ValueError('robometer_output_frame_alignment_mismatch')
            if t.shape != a.shape[:2]:
                raise ValueError('robometer_success_output_frame_alignment_mismatch')
            # Only the final visible frame is the target of this causal sample.
            probability = progress_probability(a[:, -1, :]).clamp(1e-6, 1 - 1e-6)
            for i, original in enumerate(group['indices'].tolist()):
                rows[original] = (torch.logit(probability[i]), t[i, -1].float())
        if any(row is None for row in rows):
            raise ValueError('robometer_batch_indices_incomplete')
        output = {'A': torch.stack([r[0] for r in rows]), 'T': torch.stack([r[1] for r in rows])}
        # Match the shared A/T loss API. No auxiliary Robometer heads are trained.
        for key in ('failure_actions', 'failure_types', 'failure_results', 'recovery'):
            output[key] = output['A'].new_zeros((batch_size, len(self.ontology[key])))
        return output

    def save_weights(self, folder, config):
        from safetensors.torch import save_file
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        self.rbm.model.save_pretrained(folder / 'adapter', safe_serialization=True)
        save_file({n: p.detach().cpu().contiguous() for n, p in self.heads.state_dict().items()},
                  str(folder / 'heads.safetensors'))
        (folder / 'reward_model.json').write_text(json.dumps({'config': config, 'ontology': self.ontology}, indent=2))

    def load_weights(self, folder):
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        from safetensors.torch import load_file
        folder = Path(folder)
        metadata = json.loads((folder / 'reward_model.json').read_text())
        if metadata['ontology']['sha256'] != self.ontology['sha256']:
            raise ValueError('checkpoint_ontology_mismatch')
        set_peft_model_state_dict(self.rbm.model, load_peft_weights(str(folder / 'adapter')))
        self.heads.load_state_dict(load_file(str(folder / 'heads.safetensors')), strict=True)


def build_model(config, ontology, processor):
    import yaml
    from transformers import AutoConfig
    from peft import LoraConfig, TaskType, get_peft_model
    from robometer.models.rbm import RBM
    checkpoint = Path(config['model_name_or_path']).resolve()
    native = yaml.safe_load((checkpoint / 'config.yaml').read_text())['model']
    if not (native['use_multi_image'] and native['use_per_frame_progress_token']
            and native['progress_loss_type'] == 'discrete' and not native['use_peft']):
        raise ValueError('unsupported_robometer_checkpoint_architecture')
    for token in ('<|split_token|>', '<|reward_token|>', '<|pref_token|>', '<|sim_token|>', '<|prog_token|>'):
        if token not in processor.tokenizer.get_vocab():
            raise ValueError('robometer_processor_missing_trained_special_token:' + token)
    # Direct native RBM loading avoids Unsloth and a second, unneeded 4B base load.
    # Strict loading prevents accidentally training a randomly initialized head.
    pretrained = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True)
    index = checkpoint / 'model.safetensors.index.json'
    if index.exists():
        keys = json.loads(index.read_text())['weight_map']
    else:
        from safetensors import safe_open
        with safe_open(str(checkpoint / 'model.safetensors'), framework='pt') as state:
            keys = list(state.keys())
    loader = RBM
    if any(key.startswith('similarity_head.') for key in keys):
        class LegacySimilarityRBM(RBM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Published RFM checkpoints include this retired, unused head.
                # Recreate its architecture and load its weights strictly too.
                self.similarity_head = copy.deepcopy(self.preference_head)
        loader = LegacySimilarityRBM
    rbm, info = loader.from_pretrained(str(checkpoint), config=pretrained, processor=processor, tokenizer=processor.tokenizer,
        base_model_id=native['base_model_id'], model_config=SimpleNamespace(**native),
        dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('robometer_weights_not_fully_loaded:' + json.dumps(info)[:2000])
    if len(processor.tokenizer) != rbm.model.get_input_embeddings().weight.shape[0]:
        raise ValueError('robometer_tokenizer_embedding_size_mismatch')
    rbm.requires_grad_(False)
    rbm.model = get_peft_model(rbm.model, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
        r=config['lora_rank'], lora_alpha=config['lora_alpha'], lora_dropout=0.,
        target_modules=r'language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'))
    rbm.progress_head.float().requires_grad_(True)
    rbm.success_head.float().requires_grad_(True)
    if config.get('gradient_checkpointing', True):
        rbm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = RobometerAT(rbm, ontology)
    if any(p.requires_grad for n, p in model.named_parameters() if '.visual.' in n or 'preference_head' in n or 'similarity_head' in n):
        raise ValueError('vision_and_unused_preference_head_must_be_frozen')
    model.pretrained_loading_info = {'architecture': 'RBM', **info}
    return model
