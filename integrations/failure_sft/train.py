"""Matched SFT pilots using the native OpenPI transformations and loss."""
import argparse
from contextlib import nullcontext
import dataclasses
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
from common import ROOT,SOURCE,DATA,CHECKPOINT,STATS,read,write,sha,now

def global_anchor_schedule(indices, phases, multipliers, steps, batch, seed):
    probability=np.asarray(multipliers,dtype=np.float64)[phases]
    if np.any(probability<=0) or not np.isfinite(probability).all():raise ValueError('invalid_sampling_weights')
    probability/=probability.sum()
    rng=np.random.default_rng(seed)
    return rng.choice(indices,size=steps*batch,replace=True,p=probability)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',choices=['guided','random'],required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--run',required=True);parser.add_argument('--cpu-data-check',action='store_true')
    args=parser.parse_args();run=Path(args.run);plan=read(ROOT/'plan.json')
    if args.task not in plan['tasks']:raise ValueError('unplanned_task')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from omegaconf import OmegaConf
    from rlinf.data.lerobot_paths import explicit_local_lerobot_data_files
    from rlinf.data.lerobot_workers import prepare_lerobot_dataloader_for_spawn
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    import openpi.training.data_loader as data_loader
    seed=plan['training_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if args.cpu_data_check:
        rank=0;world=1;local=0
    else:
        rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local)
        dist.init_process_group('nccl')
    config=get_openpi_config('pi05_dualfranka_robotwin_head',model_path=str(CHECKPOINT),
        norm_stats_source_dir=str(STATS),repo_id=str(DATA),batch_size=world)
    config=dataclasses.replace(config,num_workers=0 if args.cpu_data_check else 2,seed=seed)
    with explicit_local_lerobot_data_files(str(DATA)):
        loader=data_loader.create_data_loader(config,framework='pytorch',shuffle=False)
    inner=loader._data_loader.torch_loader
    prepare_lerobot_dataloader_for_spawn(inner,max_open_shards=4)
    from clip_training import ClipDataset,batches,install_masked_sft
    anchors=np.load(ROOT/'datasets'/args.task/'anchors.npz')
    indices,valid=anchors[args.group],anchors['valid_steps']
    dataset=ClipDataset(inner.dataset,indices,valid)
    if args.cpu_data_check:
        for index in sorted(set([0,len(indices)//2,len(indices)-1,int(np.argmin(valid)),int(np.argmax(valid))])):
            sample=dataset[index]
            assert np.asarray(sample['actions']).shape==(50,32)
            assert np.isfinite(np.asarray(sample['actions'])).all()
            assert np.array_equal(np.asarray(sample['actions']),np.asarray(inner.dataset[int(indices[index])]['actions'],dtype=np.float32))
            assert int(sample['clip_loss_mask'].sum())==int(valid[index])
        probe=torch.utils.data.DataLoader(dataset,batch_size=1,sampler=[0,len(dataset)//2],num_workers=1,
            collate_fn=inner.collate_fn,worker_init_fn=inner.worker_init_fn,multiprocessing_context='spawn')
        probe_batch=next(batches(probe))
        assert probe_batch['clip_loss_mask'].shape[1]==50
        print(json.dumps({'CPU_data_check':True,'anchors':len(indices),'action_shape':[50,32],
                          'outside_clip_real_targets_preserved':True}));return
    schedule=np.random.default_rng(seed+10000).choice(len(dataset),size=plan['train_steps']*plan['global_batch'],replace=True)
    class FixedSampler:
        def __iter__(self):return iter(schedule[rank::world].tolist())
        def __len__(self):return len(schedule[rank::world])
    selected_loader=torch.utils.data.DataLoader(dataset,batch_size=1,sampler=FixedSampler(),num_workers=2,
        collate_fn=inner.collate_fn,worker_init_fn=inner.worker_init_fn,
        multiprocessing_context='spawn',persistent_workers=True)
    out=run/args.group
    if rank==0:
        out.mkdir(parents=True,exist_ok=False)
        np.save(out/'anchor_schedule.npy',schedule)
        write(out/'run.json',{'group':args.group,'task':args.task,'started_at':now(),'plan_sha256':sha(ROOT/'plan.json'),
             'world_size':world,'selected_clips_only':True,'outside_actions_preserved':True,'steps':plan['train_steps'],
             'schedule_sha256':sha(out/'anchor_schedule.npy'),'checkpoint':str(CHECKPOINT)})
    dist.barrier()
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.models.embodiment.base_policy import ForwardType
    cfg=OmegaConf.create({'model_path':str(CHECKPOINT),'openpi':{
        'config_name':'pi05_dualfranka_robotwin_head','norm_stats_source_dir':str(STATS),
        'strict_checkpoint':True,'action_horizon':50,'action_chunk':50,'action_env_dim':16,
        'action_dim':32,'num_images_in_input':3,'train_expert_only':False,'add_value_head':False,
        'num_steps':10,'use_dsrl':False,'detach_critic_input':True,'noise_method':'flow_sde'}})
    model=get_model(cfg).to(torch.device('cuda',local))
    install_masked_sft(model)
    model.train()
    ddp=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['lr'],
        betas=(0.9,0.95),eps=1e-8,weight_decay=1e-10)
    # Model initialization is identical; flow noise and augmentation are distinct
    # across ranks but identically seeded between the two experimental groups.
    random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
    accumulation=plan['global_batch']//world
    if accumulation*world!=plan['global_batch']:raise ValueError('global_batch_not_divisible')
    iterator=batches(selected_loader);clock=time.monotonic()
    if rank==0:print(json.dumps({'event':'training_ready','task':args.task,'group':args.group,'world_size':world,'outside_actions_preserved':True}),flush=True)
    for step in range(1,plan['train_steps']+1):
        warmup=plan['warmup_steps']
        lr=(plan['lr']*step/warmup if step<=warmup else plan['min_lr']+(plan['lr']-plan['min_lr'])*
            (1+math.cos(math.pi*(step-warmup)/(plan['train_steps']-warmup)))/2)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);mean_loss=torch.zeros((),device=local)
        for micro in range(accumulation):
            batch=next(iterator)
            with ddp.no_sync() if micro+1<accumulation else nullcontext():
                loss=ddp(forward_type=ForwardType.SFT,data=batch)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite_training_loss')
                (loss/accumulation).backward();mean_loss+=loss.detach()/accumulation
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();dist.all_reduce(mean_loss);mean_loss/=world
        if rank==0:
            record={'step':step,'budget':plan['train_steps'],'loss':float(mean_loss),'grad_norm':float(grad),
                    'lr':lr,'elapsed_seconds':time.monotonic()-clock,'at':now()}
            print(json.dumps(record),flush=True);write(out/'status.json',record)
        if step in plan['save_steps']:
            dist.barrier()
            if rank==0:
                cp=out/'checkpoints'/('global_step_'+str(step));weights=cp/'actor/model_state_dict'
                weights.mkdir(parents=True)
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},weights/'full_weights.pt')
                write(cp/'checkpoint.json',{'step':step,'source_checkpoint':str(CHECKPOINT),
                     'weights_sha256':sha(weights/'full_weights.pt'),'optimizer_resume_available':False})
            dist.barrier()
    if rank==0:write(out/'complete.json',{'steps':plan['train_steps'],'finished_at':now()})
    dist.destroy_process_group()

if __name__=='__main__':main()
