"""Read-only checks; this module never builds a model or starts training."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from qwen_rm.data import SampleDataset
from qwen_rm.train import load_config


def inspect(config_path, snapshot):
    config = load_config(config_path)
    snapshot = Path(snapshot).resolve()
    report = json.loads((snapshot / 'snapshot.json').read_text())
    if report.get('history_max_frames') != 8 or config.get('history_max_frames') != 8:
        raise ValueError('this_experiment_requires_frozen_history8_snapshot')
    if not report.get('test_is_never_training_data'):
        raise ValueError('snapshot_split_contract_missing')
    ontology = json.loads((snapshot / 'ontology.json').read_text())
    if ontology['sha256'] != report['ontology_sha256']:
        raise ValueError('snapshot_ontology_changed')
    cases, summary = {}, {}
    for split in ('train', 'validation', 'test'):
        dataset = SampleDataset(snapshot, split, config['mode'])
        if not dataset.rows:
            raise ValueError('empty_split:' + split)
        ids = [r['id'] for r in dataset.rows]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate_score_point_ids:' + split)
        cases[split] = {r['case_id'] for r in dataset.rows}
        terminal = [r for r in dataset.rows if r['targets']['T']['mask']]
        if len(terminal) != len(cases[split]):
            raise ValueError('exactly_one_terminal_T_required_per_case:' + split)
        summary[split] = {'trajectories': len(cases[split]), 'score_points': len(dataset),
            'known_A_points': sum(r['targets']['A']['mask'] for r in dataset.rows),
            'terminal_outcomes': dict(Counter('success' if r['targets']['T']['value'] else 'failure' for r in terminal)),
            'tasks': dict(Counter(r['task'] for r in terminal))}
    if any(cases[a] & cases[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('trajectory_split_leakage')
    from .runtime import source_fingerprint
    result = {'backend': config['backend'], 'method': config['method'], 'splits': summary,
        'snapshot_sha256': hashlib.sha256((snapshot / 'snapshot.json').read_bytes()).hexdigest(),
        'effective_batch': config['batch_size'] * config['gradient_accumulation_steps'] * config['required_world_size'],
        'world_size': config['required_world_size'], 'history_max_frames': 8,
        'source_sha256': source_fingerprint(), 'training_started': False}
    if config['method'] == 'native_sft':
        from native_rm.data import NativeDataset
        result['native_sft_training_data'] = {}
        for split in ('train', 'validation'):
            dataset = NativeDataset(snapshot, split, config)
            if not len(dataset):
                raise ValueError('no_known_native_training_answers:' + split)
            result['native_sft_training_data'][split] = {'score_points': len(dataset), 'skipped': dict(dataset.skipped)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--snapshot', required=True)
    p.add_argument('--check-gpus', action='store_true'); p.add_argument('--output')
    args = p.parse_args()
    result = inspect(args.config, args.snapshot)
    if args.check_gpus:
        import torch
        count = torch.cuda.device_count()
        if count != result['world_size']:
            raise ValueError('expected_exactly_eight_visible_GPUs')
        result['visible_gpu_count'] = count
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with Path(args.output).open('x', encoding='utf-8') as f:
            f.write(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
