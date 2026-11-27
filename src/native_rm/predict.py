from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from qwen_rm.checkpoints import model_fingerprint, validate_calibration
from qwen_rm.metrics import auc, choose_threshold, classification
from qwen_rm.data import SampleDataset
from dual_rm.runtime import build_runtime
from .contracts import parse_native


def metrics(records, split, calibration=None, native_target='A_only'):
    valid = [r for r in records if r['status'] == 'ready']
    terminals = [r for r in records if r['T_mask']]
    scored_terminals = [r for r in valid if r['T_mask']]
    pairs = [(r['score'], int(r['T_target'])) for r in scored_terminals]
    if split == 'test' and calibration is None:
        raise ValueError('native_test_requires_validation_calibration')
    threshold = choose_threshold(pairs, 'validation') if split == 'validation' else calibration['score_threshold']
    errors = [abs(r['score'] - r['A_target']) for r in valid if r['A_mask']] if native_target == 'A_only' else []
    return {'split': split, 'native_scalar_is_not_a_separate_T_head': True,
        'total_score_points': len(records), 'valid_score_points': len(valid),
        'invalid_score_points': len(records) - len(valid),
        'total_terminal_cases': len(terminals), 'valid_terminal_cases': len(scored_terminals),
        'invalid_terminal_cases': len(terminals) - len(scored_terminals),
        'terminal_coverage_by_outcome': dict(Counter('success' if r['T_target'] else 'failure' for r in scored_terminals)),
        'native_training_target': native_target,
        'A_reference': 'Gemini candidate physical progress' if native_target == 'A_only' else None,
        'A_valid_points': len(errors), 'A_score_mae': sum(errors) / len(errors) if errors else None,
        'score_auc': auc(pairs), 'score_threshold': threshold, 'threshold_source': 'validation',
        'classification_scope': 'valid native terminal predictions; invalid coverage reported separately',
        **classification(pairs, threshold)}


def main():
    p = argparse.ArgumentParser(description='Generate native reward outputs; calibrate only on validation.')
    p.add_argument('--checkpoint', required=True); p.add_argument('--snapshot', required=True)
    p.add_argument('--split', choices=('validation', 'test'), required=True)
    p.add_argument('--frame-cache', required=True); p.add_argument('--output', required=True)
    p.add_argument('--calibration'); p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    import torch
    from accelerate.utils import send_to_device
    checkpoint, snapshot, out = Path(args.checkpoint), Path(args.snapshot), Path(args.output)
    metadata = json.loads((checkpoint / 'reward_model.json').read_text())
    config = metadata['config']
    if config['method'] != 'native_sft':
        raise ValueError('native_prediction_requires_native_SFT_checkpoint')
    position = json.loads((checkpoint / 'training_position.json').read_text())
    snapshot_hash = hashlib.sha256((snapshot / 'snapshot.json').read_bytes()).hexdigest()
    if snapshot_hash != position['contract']['snapshot_sha256']:
        raise ValueError('evaluation_snapshot_changed')
    binding = {'reference_snapshot_sha256': snapshot_hash, 'checkpoint_sha256': model_fingerprint(checkpoint),
               'checkpoint_step': position['step'], 'input_mode': config['mode'], 'method': 'native_sft'}
    calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
    if args.split == 'test':
        validate_calibration(calibration, binding)
    if out.exists() or out.with_suffix(out.suffix + '.metrics.json').exists():
        raise ValueError('native_prediction_output_exists')
    model, processor, collator = build_runtime(config, metadata['ontology'], args.frame_cache, checkpoint / 'processor')
    model.load_weights(checkpoint); model.to(args.device); model.eval()
    dataset = SampleDataset(snapshot, args.split, config['mode'])
    out.parent.mkdir(parents=True, exist_ok=True); records = []
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'no': torch.float32}[config['mixed_precision']]
    with out.open('x', encoding='utf-8') as stream, torch.inference_mode():
        for row in dataset.rows:
            inputs = send_to_device(collator.inference([row]), args.device)
            with torch.autocast(device_type=torch.device(args.device).type, dtype=dtype, enabled=config['mixed_precision'] != 'no'):
                generated = model.generator.generate(**inputs, max_new_tokens=config.get('max_new_tokens', 32), do_sample=False)
            raw = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
            record = {'id': row['id'], 'case_id': row['case_id'], 'task': row['task'], 'frame': row['frame'],
                'raw_text': raw, 'A_mask': row['targets']['A']['mask'], 'A_target': row['targets']['A']['value'],
                'T_mask': row['targets']['T']['mask'], 'T_target': row['targets']['T']['value']}
            try:
                record.update(status='ready', **parse_native(raw, config))
            except ValueError as error:
                record.update(status='invalid_output', error=str(error))
            records.append(record); stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    report = metrics(records, args.split, calibration, config.get('native_target', 'A_only'))
    report.update(binding)
    out.with_suffix(out.suffix + '.metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
