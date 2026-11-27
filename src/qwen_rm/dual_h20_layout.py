"""Single-GPU to two-GPU fixed-budget RM handoff."""
import hashlib,json,math,random
from pathlib import Path
from torch.utils.data import BatchSampler, RandomSampler
from .checkpoints import model_fingerprint


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def verify_parent(config, position, snapshot_sha, per_epoch):
    expected={**config,'required_world_size':1}
    expected.pop('layout_handoff',None)
    prior=position['contract']
    if (prior['world_size']!=1 or prior['snapshot_sha256']!=snapshot_sha
        or prior['config']!=expected or prior['max_steps_override']!=0 or prior['max_samples']!=0
        or position['batch_offset']!=0 or position['epoch']*per_epoch!=position['step']):
        raise ValueError('incompatible fixed-epoch checkpoint for dual-GPU migration')


class TrainingRows:
    def __init__(self, dataset):
        self.dataset=dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, entry):
        index,valid,scale=entry
        return self.dataset[index],valid,scale


class WholeMicrobatch(BatchSampler):
    """Keep the original shuffle and global microbatch membership."""
    def __iter__(self):
        for group in super().__iter__():
            count=len(group)
            scale=self.batch_size/count
            yield [(index,True,scale) for index in group]+[(group[-1],False,scale)]*(self.batch_size-count)


def training_collator(collator):
    def wrapped(items):
        import torch
        rows, valid, scales=zip(*items,strict=True)
        value=collator(rows)
        if len(set(scales))!=1:
            raise RuntimeError('microbatch split produced inconsistent tail scaling')
        value['layout_loss_scale']=scales[0]
        if not all(valid):
            for key,mask in value['targets'].items():
                if key.endswith('_mask'):
                    mask[~torch.tensor(valid,dtype=torch.bool)]=False
        return value
    return wrapped


def train_loader(original, collator, generator, batch_size):
    from torch.utils.data import DataLoader
    dataset=TrainingRows(original)
    sampler=WholeMicrobatch(RandomSampler(dataset,generator=generator),batch_size,drop_last=False)
    return DataLoader(dataset,batch_sampler=sampler,generator=generator,collate_fn=training_collator(collator),num_workers=0)


def restore(checkpoint, optimizer, scheduler, position, handoff, rank):
    import numpy as np
    import torch
    for name in ('optimizer.bin','scheduler.bin','random_states_0.pkl'):
        if digest(checkpoint/name)!=handoff['state_sha256'][name]:raise ValueError(f'checkpoint state changed: {name}')
    state=torch.load(checkpoint/'optimizer.bin',map_location='cpu',weights_only=True)
    schedule=torch.load(checkpoint/'scheduler.bin',map_location='cpu',weights_only=True)
    if schedule['last_epoch']!=position['step'] or [x['lr'] for x in state['param_groups']]!=schedule['_last_lr']:
        raise ValueError('optimizer/scheduler checkpoint step mismatch')
    slots=0
    for saved,live in zip(state['param_groups'],optimizer.param_groups,strict=True):
        for key in ('betas','eps','weight_decay','amsgrad'):
            if saved[key]!=live[key]:raise ValueError(f'optimizer group changed: {key}')
        for index,param in zip(saved['params'],live['params'],strict=True):
            slot=state['state'][index]
            if slot['exp_avg'].shape!=param.shape or slot['exp_avg_sq'].shape!=param.shape or not 0<int(slot['step'])<=position['step']:
                raise ValueError('optimizer slot mismatch')
            slots+=1
    if slots!=len(state['state']):raise ValueError('incomplete optimizer restore')
    optimizer.load_state_dict(state)
    scheduler.load_state_dict(schedule)
    # This is a local, hash-pinned Accelerate RNG checkpoint containing NumPy state.
    rng=torch.load(checkpoint/'random_states_0.pkl',map_location='cpu',weights_only=False)
    random.setstate(rng['random_state']); np.random.set_state(rng['numpy_random_seed']); torch.random.set_rng_state(rng['torch_manual_seed'])
    if rank==0:
        torch.cuda.set_rng_state(rng['torch_cuda_manual_seed'][0],device=torch.cuda.current_device())
    else:
        torch.cuda.manual_seed(42+position['step']+rank)
    return {'restored_step':position['step'],'restored_epoch':position['epoch'],'optimizer_slots':slots,'scheduler_step':schedule['last_epoch'],'rank1_rng':'derived_from_seed_step_and_rank'}
