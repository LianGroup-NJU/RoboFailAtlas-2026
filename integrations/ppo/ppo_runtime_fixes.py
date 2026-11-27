"""Private, version-pinned PPO corrections; never edits the shared RLinf checkout."""
from __future__ import annotations
import contextvars,functools,hashlib,inspect,json,os,textwrap
from pathlib import Path
import torch

_INSTALLED=False
_POOL=contextvars.ContextVar('ppo_critic_moments',default=None)
VERSION='ppo-critic-finite400-v1'

def _replace_method(method,replacements):
    original=inspect.unwrap(method)
    source=textwrap.dedent(inspect.getsource(original))
    for old,new in replacements:
        if source.count(old)!=1:
            raise RuntimeError('unsupported_RLinf_method:'+method.__name__)
        source=source.replace(old,new,1)
    namespace={}
    exec(compile('from __future__ import annotations\n'+source,
                 __file__+':'+method.__name__,'exec'),original.__globals__,namespace)
    updated=namespace[method.__name__]
    updated.__qualname__=method.__qualname__
    return updated

def check_source_fingerprints(expected):
    for path,digest in expected.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('PPO_fix_source_version_changed:'+path)

def prediction_moments(values,returns,prev_values,value_clip,loss_mask=None):
    """Additive FP64 moments, weighted by valid samples rather than microbatches."""
    v=values.detach().double().reshape(-1)
    r=returns.detach().double().reshape(-1)
    old=prev_values.detach().double().reshape(-1)
    selected=torch.ones_like(v,dtype=torch.bool) if loss_mask is None else loss_mask.detach().reshape(-1).bool().expand_as(v)
    finite=selected&torch.isfinite(v)&torch.isfinite(r)&torch.isfinite(old)
    vv=v[finite];rr=r[finite];oo=old[finite]
    err=rr-vv
    return torch.stack((selected.sum().double(),finite.sum().double(),rr.sum(),vv.sum(),
                        rr.square().sum(),vv.square().sum(),err.square().sum(),
                        ((vv-oo).abs()>value_clip).sum().double(),err.abs().sum()))

def metrics_from_moments(m):
    nsel,n,sr,sv,srr,svv,se2,nclip,sae=[float(x)for x in m]
    if n<=0:
        return {'critic/explained_variance':float('nan'),'critic/pooled_samples':0.,
                'critic/nonfinite_fraction':1. if nsel else 0.}
    mr=sr/n;mv=sv/n
    vr=max(0.,srr/n-mr*mr);vv=max(0.,svv/n-mv*mv)
    ve=max(0.,se2/n-(mr-mv)**2)
    ev=1.-ve/vr if n>=2 and vr>1e-12 else float('nan')
    return {'critic/explained_variance':ev,'critic/return_mean':mr,'critic/return_std':vr**.5,
            'critic/value_mean':mv,'critic/value_std':vv**.5,'critic/value_mse':se2/n,
            'critic/value_mae':sae/n,'critic/residual_variance':ve,
            'critic/value_clip_ratio':nclip/n,'critic/pooled_samples':n,
            'critic/nonfinite_fraction':(nsel-n)/nsel if nsel else 0.}

def install(*,model_cls,loss_module,actor_cls,env_cls,expected_sources):
    global _INSTALLED
    if _INSTALLED:return
    check_source_fingerprints(expected_sources)
    sample_method=_replace_method(model_cls.sample_actions,[(
        'values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)',
        'values = torch.stack(values, dim=1)\n'
        '        if not self.config.joint_logprob and mode == "train":\n'
        '            selected = denoise_inds[:, 0].to(values.device)\n'
        '            values = values[torch.arange(bsize, device=values.device), selected][:, None]\n'
        '        else:\n'
        '            values = values.mean(dim=-1, keepdim=True)')])
    vlm_method=_replace_method(model_cls.get_value_from_vlm,[(
        'prefix_out_value = prefix_out_value.to(dtype=torch.float32)',
        'prefix_out_value = prefix_out_value.to(dtype=torch.float32)\n'
        '    if self.config.detach_critic_input:\n'
        '        prefix_out_value = prefix_out_value.detach()')])
    # Replacements below use indentation from the original method after dedent.
    fixed_loss=_replace_method(loss_module.compute_ppo_critic_loss,[(
        '(value_pred_clipped - prev_values).abs() > value_clip',
        '(values - prev_values).abs() > value_clip')])
    loss_signature=inspect.signature(fixed_loss)
    @functools.wraps(fixed_loss)
    def loss(*args,**kwargs):
        answer=fixed_loss(*args,**kwargs)
        pool=_POOL.get()
        if pool is not None:
            bound=loss_signature.bind(*args,**kwargs)
            bound.apply_defaults();k=bound.arguments
            with torch.no_grad():
                moments=prediction_moments(k['values'],k['returns'],k['prev_values'],
                                           k['value_clip'],k.get('loss_mask'))
            pool.append(moments)
        return answer
    original_train=actor_cls.run_training
    @functools.wraps(original_train)
    def train(self,*args,**kwargs):
        pool=[];token=_POOL.set(pool)
        try:
            metrics=original_train(self,*args,**kwargs)
        finally:
            _POOL.reset(token)
        if not pool:raise RuntimeError('PPO_critic_moments_not_collected')
        moments=torch.stack(pool).sum(dim=0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(moments,op=torch.distributed.ReduceOp.SUM)
        metrics.update(metrics_from_moments(moments.cpu().tolist()))
        return metrics
    env_method=_replace_method(env_cls.chunk_step,[
       ('terminal[i] = row["success"]',
        'terminal[i] = row["done"] if self.cfg.get("task_horizon_is_terminal", False) else row["success"]'),
       ('truncated[i] = row["done"] and not row["success"]',
        'truncated[i] = row["done"] and not bool(terminal[i])')])
    model_cls.sample_actions=sample_method
    model_cls.get_value_from_vlm=vlm_method
    loss_module.compute_ppo_critic_loss=loss
    actor_cls.run_training=train
    env_cls.chunk_step=env_method
    _INSTALLED=True
    print(json.dumps({'event':'ppo_critic_corrections_installed','version':VERSION,
                      'finite_task_horizon_requires_env_flag':True,
                      'pooled_metrics_scope':'all finite valid training predictions across ranks and update epochs',
                      'shared_checkout_modified':False}),flush=True)
