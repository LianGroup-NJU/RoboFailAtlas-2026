from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"model_name_or_path", "mode", "method", "mixed_precision", "lora_rank", "lora_alpha", "seed",
                "batch_size", "gradient_accumulation_steps", "learning_rate", "head_learning_rate", "num_epochs",
                "warmup_ratio", "weight_decay", "max_pixels", "min_pixels", "max_tokens", "save_every", "eval_every", "loss_weights"}
    if not required <= set(config) or config["mode"] not in ("single", "history") or config["method"] not in ("at", "joint", "native_sft"):
        raise ValueError("invalid_training_config")
    if config["mode"] == "single" and config["method"] == "joint":
        raise ValueError("trajectory_failure_history_requires_history_input")
    if config["mixed_precision"] not in ("bf16", "fp16", "no"):
        raise ValueError("invalid_mixed_precision")
    if config.get('backend') not in ('qwen3vl', 'robometer', 'densereward', 'roboreward'):
        raise ValueError('invalid_backend')
    if config['backend'] != 'qwen3vl' and config['method'] not in ('at', 'native_sft'):
        raise ValueError('public_RM_requires_at_or_native_sft')
    if config['method'] == 'native_sft' and config['backend'] not in ('densereward', 'roboreward'):
        raise ValueError('native_SFT_requires_generative_public_RM')
    if config['method'] in ('at', 'joint') and config.get('score_losses') != {'A': 'mse', 'T': 'bce_with_logits'}:
        raise ValueError('current_score_head_training_requires_explicit_MSE_and_BCE_losses')
    if any(type(config[k]) is not int or config[k] <= 0 for k in ('batch_size', 'gradient_accumulation_steps', 'required_world_size')):
        raise ValueError('invalid_batch_or_world_size')
    return config


def evaluation_records(logits, target):
    n = len(logits["A"])
    return [{"A_prediction": logits["A"][i], "T_prediction": logits["T"][i],
             "A_target": target["A_value"][i], "T_target": target["T_value"][i],
             "A_mask": target["A_mask"][i], "T_mask": target["T_mask"][i]} for i in range(n)]


def initialize_run(out, contract, resume, accelerator):
    from accelerate.utils import broadcast_object_list
    failure = [None]
    if accelerator.is_main_process:
        try:
            out.mkdir(parents=True, exist_ok=True)
            path = out / 'run.json'
            if path.exists():
                if not resume or json.loads(path.read_text(encoding='utf-8')) != contract:
                    raise ValueError('output_already_has_a_different_run_use_resume_or_a_new_directory')
            else:
                with path.open('x', encoding='utf-8') as stream:
                    json.dump(contract, stream, ensure_ascii=False, indent=2)
        except (OSError, ValueError) as error:
            failure[0] = str(error)
    broadcast_object_list(failure, from_process=0)
    if failure[0]:
        raise ValueError(failure[0])
    accelerator.wait_for_everyone()


def require_new_checkpoint(directory, accelerator):
    from accelerate.utils import broadcast_object_list
    exists = [directory.exists() if accelerator.is_main_process else None]
    broadcast_object_list(exists, from_process=0)
    if exists[0]:
        raise ValueError('checkpoint_would_overwrite_existing_state')


def evaluate(model, loader, accelerator, method='at'):
    if method == 'native_sft':
        from native_rm.model import evaluate as native_evaluate
        return native_evaluate(model, loader, accelerator)
    import torch
    from collections import defaultdict
    from .metrics import summarize,auc,classification
    from .structured_metrics import StructuredCounts
    ontology=json.loads((loader.dataset.snapshot/'ontology.json').read_text())
    tasks=sorted({r['task'] for r in loader.dataset.rows});task_ids={t:i for i,t in enumerate(tasks)}
    total=StructuredCounts(ontology);by_task={t:StructuredCounts(ontology) for t in tasks}
    records=[];task_records=defaultdict(list);model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits=model(**batch['inputs'])
            data={'A':logits['A'].float().sigmoid(),'T':logits['T'].float().sigmoid(),
                  **{k:batch['targets'][k] for k in ('A_value','T_value','A_mask','T_mask')},
                  'task_id':torch.tensor([task_ids[c.split('/')[0]] for c in batch['case_ids']],device=logits['A'].device),
                  'recovery_pred':logits['recovery'].argmax(-1),'recovery_value':batch['targets']['recovery_value'],
                  'recovery_mask':batch['targets']['recovery_mask']}
            for head in ('failure_actions','failure_types','failure_results'):
                data.update({head+'_pred':logits[head].float().sigmoid(),
                             head+'_value':batch['targets'][head+'_value'],head+'_mask':batch['targets'][head+'_mask']})
            data=accelerator.gather_for_metrics(data)
            values={k:v.cpu().tolist() for k,v in data.items()}
            rows=evaluation_records(values,values);records+=rows
            for i,row in enumerate(rows):
                task=tasks[values['task_id'][i]];task_records[task].append(row)
                total.add(values,i);by_task[task].add(values,i)
    report=summarize(records,'validation');report['structured']=total.summary();report['per_task']={}
    for task,rows in task_records.items():
        errors=[abs(r['A_prediction']-r['A_target']) for r in rows if r['A_mask']]
        pairs=[(r['T_prediction'],int(r['T_target'])) for r in rows if r['T_mask']]
        report['per_task'][task]={'samples':len(rows),'A_valid_points':len(errors),'A_mae':sum(errors)/len(errors) if errors else None,
            'T_terminal_cases':len(pairs),'T_auc':auc(pairs),'T_threshold':report['T_threshold'],
            'threshold_source':'full_validation_global_threshold',**classification(pairs,report['T_threshold']),
            'structured':by_task[task].summary()}
    model.train()
    return report


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--snapshot", required=True)
    p.add_argument("--output", required=True); p.add_argument("--frame-cache", required=True)
    p.add_argument("--resume"); p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0); p.add_argument("--stop-after-steps", type=int, default=0)
    args = p.parse_args(argv)
    from dual_rm.runtime import require_training_released, build_runtime, parameter_groups, source_fingerprint
    require_training_released()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from accelerate import Accelerator
    from accelerate.utils import DataLoaderConfiguration, set_seed
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    from .data import SampleDataset
    from .model import loss_fn
    from .checkpoints import model_fingerprint
    from .convergence import Convergence
    from .dual_h20_layout import digest, verify_parent, restore, train_loader as make_split_train_loader
    import datetime
    config = load_config(args.config); snapshot = Path(args.snapshot).resolve(); out = Path(args.output).resolve()
    accelerator = Accelerator(mixed_precision=config["mixed_precision"], gradient_accumulation_steps=config["gradient_accumulation_steps"],
                              step_scheduler_with_optimizer=False,
                              dataloader_config=DataLoaderConfiguration(split_batches=config['required_world_size']>1))
    if accelerator.num_processes != config['required_world_size']:
        raise ValueError('world_size_does_not_match_frozen_experiment_config')
    set_seed(config["seed"])
    ontology = json.loads((snapshot / "ontology.json").read_text(encoding="utf-8"))
    snapshot_hash = hashlib.sha256((snapshot / "snapshot.json").read_bytes()).hexdigest()
    if config['method'] == 'native_sft':
        from native_rm.data import NativeDataset
        train = NativeDataset(snapshot, 'train', config, args.max_samples)
        validation = NativeDataset(snapshot, 'validation', config, args.max_samples)
        config['native_dataset_skips'] = {'train': dict(train.skipped), 'validation': dict(validation.skipped)}
    else:
        train = SampleDataset(snapshot, "train", config["mode"], args.max_samples)
        validation = SampleDataset(snapshot, "validation", config["mode"], args.max_samples)
    if config.get('history_max_frames', 16) != train.history_max_frames or train.history_max_frames != validation.history_max_frames:
        raise ValueError('training_config_snapshot_history_budget_mismatch')
    if not len(train) or not len(validation):
        raise ValueError("nonempty_train_and_validation_required")
    density = {key: sum(r['targets'][key]['mask'] for r in train.rows) / len(train) for key in ('A', 'T', 'recovery')}
    for key in ('failure_actions', 'failure_types', 'failure_results'):
        density[key] = sum(sum(r['targets'][key]['mask']) for r in train.rows) / (len(train) * len(ontology[key]))
    config['supervision_density'] = density
    contract = {"config": config, "snapshot": str(snapshot), "snapshot_sha256": snapshot_hash,
                "max_steps_override": args.max_steps, "max_samples": args.max_samples, "world_size": accelerator.num_processes,
                "source_sha256": source_fingerprint()}
    if args.resume:
        saved = json.loads((Path(args.resume) / "training_position.json").read_text())
        if saved["contract"] != contract:
            raise ValueError("resume_training_contract_changed")
    initialize_run(out, contract, args.resume, accelerator)
    model, processor, collator = build_runtime(config, ontology, args.frame_cache)
    handoff = config.get('layout_handoff')
    if handoff and not args.resume:
        parent = Path(handoff['checkpoint'])
        if digest(parent/'training_position.json')!=handoff['position_sha256'] or model_fingerprint(parent)!=handoff['model_sha256']:
            raise ValueError('handoff weights or position changed')
        parent_position=json.loads((parent/'training_position.json').read_text())
        model.load_weights(parent)
    generator = torch.Generator().manual_seed(config["seed"])
    if accelerator.num_processes>1:
        train_loader=make_split_train_loader(train,collator,generator,config['batch_size'])
    else:
        train_loader=DataLoader(train,batch_size=config['batch_size'],shuffle=True,generator=generator,collate_fn=collator,num_workers=0)
    valid_loader = DataLoader(validation, batch_size=config["batch_size"], shuffle=False, collate_fn=collator, num_workers=0)
    groups = parameter_groups(model, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)
    per_epoch = math.ceil(len(train_loader) / config["gradient_accumulation_steps"])
    total_steps = args.max_steps or math.ceil(config["num_epochs"] * per_epoch)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * config["warmup_ratio"]), total_steps)
    scheduler = accelerator.prepare(scheduler)

    def save_hook(models, weights, directory):
        if accelerator.is_main_process:
            accelerator.unwrap_model(models[0]).save_weights(directory, config)
            processor.save_pretrained(Path(directory) / "processor")
        weights.clear()  # frozen 4B/8B base weights are referenced, not duplicated

    def load_hook(models, directory):
        while models:
            accelerator.unwrap_model(models.pop()).load_weights(directory)

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
    step, epoch, offset = 0, 0, 0
    if args.resume:
        accelerator.load_state(args.resume)
        step, epoch, offset = saved["step"], saved["epoch"], saved["batch_offset"]
    elif handoff:
        verify_parent(config,parent_position,snapshot_hash,per_epoch)
        result=restore(parent,optimizer,scheduler,parent_position,handoff,accelerator.process_index)
        step,epoch,offset=(parent_position[k] for k in ('step','epoch','batch_offset'))
        if accelerator.is_main_process:
            (out/'layout_handoff_restored.json').write_text(json.dumps(result,indent=2)+'\n')
    last_saved = -1
    stopper = Convergence(math.ceil(config.get('convergence_min_epochs', 2) * per_epoch),
                          config.get('convergence_patience', 4), config.get('convergence_min_delta', 0.0005),
                          saved.get('convergence') if args.resume else parent_position.get('convergence') if handoff else None)
    last_report = None
    last_eval_step = -1
    def progress(phase, **extra):
        if accelerator.is_main_process:
            value = {'phase': phase, 'step': step, 'max_steps': total_steps,
                     'epoch': step / per_epoch, 'input_frame_budget': config['input_frame_budget'],
                     'updated_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                     **extra}
            temporary = out / '.status.json.tmp'
            temporary.write_text(json.dumps(value, indent=2))
            temporary.replace(out / 'status.json')
    progress('training_initializing')

    def checkpoint(next_epoch, next_offset):
        nonlocal last_saved
        if step == last_saved:
            return
        directory = out / ("checkpoint-%06d" % step)
        require_new_checkpoint(directory, accelerator)
        accelerator.save_state(str(directory))
        if accelerator.is_main_process:
            (directory / "training_position.json").write_text(json.dumps({"step": step, "epoch": next_epoch,
                    "batch_offset": next_offset, "contract": contract, "convergence": stopper.state}, indent=2))
        accelerator.wait_for_everyone(); last_saved = step

    model.train(); optimizer.zero_grad(); stop = False
    while step < total_steps and not stop:
        generator.manual_seed(config["seed"] + epoch)
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        iterator = accelerator.skip_first_batches(train_loader, offset) if offset else train_loader
        for batch_index, batch in enumerate(iterator, start=offset):
            with accelerator.accumulate(model):
                logits = model(**batch["inputs"]); loss, parts = loss_fn(logits, batch["targets"], config)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite_training_loss")
                loss=loss*batch.get('layout_loss_scale',1.0)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    norm = accelerator.clip_grad_norm_(model.parameters(), 1.)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("nonfinite_training_gradient")
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad()
            next_epoch = epoch + 1 if batch_index + 1 >= len(train_loader) else epoch
            next_offset = 0 if next_epoch != epoch else batch_index + 1
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                step += 1
                frame_counts = batch.get('input_frame_counts', [])
                stats = {"step": step, "loss": float(loss.detach()), "grad_norm": float(norm.detach()),
                         "parts": {k: float(v.detach()) for k, v in parts.items()},
                         "max_input_frames_in_last_microbatch": max(frame_counts) if frame_counts else None,
                         "sequence_tokens": int(batch['inputs']['input_ids'].shape[-1]),
                         "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()}
                accelerator.print(json.dumps(stats), flush=True)
                progress('training', latest=stats)
                if (config["eval_every"] and step % config["eval_every"] == 0) or step >= total_steps or (args.stop_after_steps and step >= args.stop_after_steps):
                    progress('full_validation')
                    report = evaluate(model, valid_loader, accelerator, config['method'])
                    if args.max_samples == 0 and (report['samples'] != 46391 or report['A_valid_points'] != 46342 or report['T_terminal_cases'] != 2941):
                        raise ValueError('incomplete_full_validation')
                    stopper.update(report, step)
                    checkpoint(next_epoch, next_offset)
                    report.update(checkpoint_step=step, reference_snapshot_sha256=snapshot_hash,
                                  ontology_sha256=ontology['sha256'], input_mode=config['mode'],
                                  method=config['method'], backend=config['backend'],
                                  checkpoint_sha256=model_fingerprint(out / ('checkpoint-%06d' % step)))
                    if accelerator.is_main_process:
                        (out / ('validation-%06d.json' % step)).write_text(json.dumps(report, indent=2))
                    accelerator.wait_for_everyone()
                    last_report = report; last_eval_step = step
                    progress('training', convergence=stopper.state)
                    accelerator.print(json.dumps({'validation_step':step,'A_mae':report['A_mae'],'T_auc':report['T_auc'],'convergence':stopper.state}),flush=True)
                    if stopper.state['converged'] and not config.get('fixed_epochs', False):
                        stop = True; break
                if (config.get('save_first_step', False) and step == 1) or (config["save_every"] and step % config["save_every"] == 0):
                    checkpoint(next_epoch, next_offset)
                if step >= total_steps or args.stop_after_steps and step >= args.stop_after_steps:
                    checkpoint(next_epoch, next_offset); stop = True; break
        epoch += 1; offset = 0
    if last_report is None or last_eval_step != step:
        raise ValueError('final_checkpoint_requires_corresponding_validation')
    fixed_epochs = config.get('fixed_epochs', False)
    if fixed_epochs and (step != total_steps or epoch != config['num_epochs']):
        raise ValueError('fixed_epochs_run_did_not_complete_requested_budget')
    best_step = step if fixed_epochs else stopper.state['best_step']
    selected = out / ('checkpoint-%06d' % best_step)
    if accelerator.is_main_process:
        report = json.loads((out / ('validation-%06d.json' % best_step)).read_text())
        (out / 'validation_calibration.json').write_text(json.dumps(report, indent=2))
        reason = 'fixed_epochs_complete' if fixed_epochs else ('validation_plateau' if stopper.state['converged'] else ('smoke_budget' if args.stop_after_steps else 'max_budget_not_converged'))
        selection = {'checkpoint':str(selected),'step':best_step,'converged':stopper.state['converged'],
                     'stop_reason':reason,'finished_steps':step,'full_data':args.max_samples==0,
                     'validation_A_mae':report['A_mae'],'validation_T_auc':report['T_auc'],
                     'rule': 'final checkpoint after all epochs' if fixed_epochs else 'minimum full-validation A MAE; exact ties by T AUC; no test selection',
                     'convergence':stopper.state}
        (out / 'selected_checkpoint.json').write_text(json.dumps(selection, indent=2))
        progress('training_converged' if stopper.state['converged'] and not fixed_epochs else reason, selection=selection)
    accelerator.print(json.dumps({'finished_steps':step,'selected_step':best_step,'convergence':stopper.state}),flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
