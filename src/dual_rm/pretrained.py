"""Strict Qwen3-VL loading shared by score-head adaptation and native SFT."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_processor(path, config):
    from transformers import AutoProcessor
    root = Path(path)
    metadata_path = root / 'tokenizer_config.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    options = dict(min_pixels=config['min_pixels'], max_pixels=config['max_pixels'],
                   local_files_only=config.get('local_files_only', True))
    extra = metadata.get('extra_special_tokens')
    if isinstance(extra, list):
        # Transformers 5.x uses a list; 4.57 expects named token aliases here.
        # These are already trained vocabulary entries, not new tokens.
        options['extra_special_tokens'] = {}
        options['additional_special_tokens'] = list(dict.fromkeys(metadata.get('additional_special_tokens', []) + extra))
    processor = AutoProcessor.from_pretrained(str(root), **options)
    if isinstance(extra, list):
        ids = {value['content']: int(key) for key,value in metadata.get('added_tokens_decoder', {}).items()}
        for token in extra:
            if token in ids and processor.tokenizer.convert_tokens_to_ids(token) != ids[token]:
                raise ValueError('compatibility_conversion_changed_trained_token_ID')
    if hasattr(processor, 'video_processor'):
        processor.video_processor.size = {'shortest_edge': config['min_pixels'], 'longest_edge': config['max_pixels']}
    return processor


def compatible_config(config):
    from transformers import AutoConfig
    root = Path(config['model_name_or_path'])
    expected = config.get('checkpoint_config_sha256')
    if expected and hashlib.sha256((root / 'config.json').read_bytes()).hexdigest() != expected:
        raise ValueError('pretrained_checkpoint_config_changed')
    model = AutoConfig.from_pretrained(str(root), local_files_only=config.get('local_files_only', True))
    if model.model_type != 'qwen3_vl':
        raise ValueError('expected_Qwen3_VL_checkpoint')
    text = model.text_config
    parameters = getattr(text, 'rope_parameters', None)
    if getattr(text, 'rope_scaling', None) is None and isinstance(parameters, dict):
        # DenseReward was exported with Transformers 5.x. Reproduce the
        # original baseline's explicit 4.57-compatible rotary configuration.
        text.rope_scaling = {k: v for k, v in parameters.items() if k != 'rope_theta'}
        if 'rope_theta' in parameters:
            text.rope_theta = parameters['rope_theta']
    return model


def load_conditional(config):
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    model_config = compatible_config(config)
    if 'Qwen3VLForConditionalGeneration' not in (model_config.architectures or []):
        raise ValueError('native_SFT_requires_conditional_generation_checkpoint')
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'no': torch.float32}[config['mixed_precision']]
    model, info = Qwen3VLForConditionalGeneration.from_pretrained(config['model_name_or_path'],
        config=model_config, dtype=dtype, attn_implementation='sdpa',
        local_files_only=config.get('local_files_only', True), output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('pretrained_weights_not_fully_loaded:' + json.dumps(info)[:2000])
    model.config.use_cache = False
    return model, info
