from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def require_training_released():
    if (ROOT / 'TRAINING_HOLD').exists():
        raise RuntimeError('training_on_hold_wait_for_explicit_user_start_instruction')


def source_fingerprint():
    files = [p for package in ('qwen_rm', 'dual_rm', 'native_rm') for p in (ROOT / package).glob('*.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def bind_robometer_source(config):
    root = Path(config['robometer_source']).resolve()
    if not (root / 'robometer/models/rbm.py').is_file():
        raise ValueError('robometer_source_missing')
    commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True).stdout.strip()
    if commit != config['robometer_commit']:
        raise ValueError('robometer_source_commit_mismatch')
    # Loading executable Python must use the exact reviewed model implementation.
    recorded = json.loads((ROOT / 'reference/upstream.json').read_text())['sha256']
    for name in ('robometer/models/rbm.py', 'robometer/models/heads.py', 'robometer/models/utils.py'):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != recorded[name]:
            raise ValueError('robometer_source_file_modified:' + name)
    sys.path.insert(0, str(root))
    if 'robometer' in sys.modules:
        location = Path(sys.modules['robometer'].__file__).resolve()
        if not location.is_relative_to(root):
            raise ValueError('another_robometer_package_is_already_imported')


def build_runtime(config, ontology, frame_cache, processor_path=None):
    from .pretrained import load_processor
    import os
    config = dict(config)
    if os.environ.get("FAILATLAS_BASE_MODEL"):
        config["model_name_or_path"] = os.environ["FAILATLAS_BASE_MODEL"]
    backend = config.get('backend', 'qwen3vl')
    processor = load_processor(processor_path or config.get('processor_name_or_path', config['model_name_or_path']), config)
    processor.tokenizer.padding_side = 'right'
    if config.get('method') == 'native_sft':
        from native_rm.model import build_model
        from native_rm.data import NativeCollator
        return build_model(config, ontology), processor, NativeCollator(processor, frame_cache, config)
    if backend in ('qwen3vl', 'densereward', 'roboreward'):
        from qwen_rm.data import Collator
        from qwen_rm.model import build_model
        from .frame_protocol import FrameBudgetCollator
        collator = FrameBudgetCollator(processor, frame_cache, config['max_tokens'], config.get('input_frame_budget', 8), config.get('frame_sampling', 'uniform_prefix'))
        return build_model(config, ontology), processor, collator
    if backend == 'robometer':
        from .robometer import build_model, RobometerCollator
        bind_robometer_source(config)
        # Published weights omit a processor. Reproduce the upstream token
        # insertion order without ever resizing or initializing model embeddings.
        for token in ('<|split_token|>', '<|reward_token|>', '<|pref_token|>', '<|sim_token|>', '<|prog_token|>'):
            if token not in processor.tokenizer.get_vocab():
                processor.tokenizer.add_special_tokens({'additional_special_tokens': [token]})
        model = build_model(config, ontology, processor)
        return model, processor, RobometerCollator(processor, frame_cache, config['max_tokens'])
    raise ValueError('unknown_backend:' + backend)


def parameter_groups(model, config):
    heads = [p for p in model.heads.parameters() if p.requires_grad]
    head_ids = {id(p) for p in heads}
    adapters = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    if config.get('method') == 'native_sft':
        if heads or not adapters:
            raise ValueError('native_SFT_requires_LoRA_without_new_score_heads')
        return [{'params': adapters, 'lr': config['learning_rate']}]
    if not heads or not adapters:
        raise ValueError('expected_trainable_adapters_and_heads')
    return [{'params': adapters, 'lr': config['learning_rate']},
            {'params': heads, 'lr': config['head_learning_rate']}]
