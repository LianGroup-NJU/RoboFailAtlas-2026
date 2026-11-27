"""Effective critic warmup, frozen-SFT transition KL, and pooled diagnostics."""
from __future__ import annotations
from pathlib import Path
import json,os
import torch
from ppo_runtime_fixes import prediction_moments,metrics_from_moments
VERSION="jointkl-rebased-oldprobs-warm10-v1"
def warmup_active(actor):
    return int(actor.version)<int(actor.cfg.algorithm.critic_warmup_rounds)
def _distributed_sum(tensor):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor,op=torch.distributed.ReduceOp.SUM)
    return tensor
def _expanded_mask(value,mask):
    selected=torch.ones_like(value,dtype=torch.bool)if mask is None else mask.bool()
    while selected.ndim<value.ndim:selected=selected.unsqueeze(-1)
    return selected.expand_as(value)
def _masked_sum_count(value,mask):
    selected=_expanded_mask(value,mask)
    if not torch.isfinite(value[selected]).all():raise FloatingPointError("nonfinite_KL_on_valid_samples")
    return torch.where(selected,value,0).sum(),selected.sum().to(value.dtype)
def gaussian_transition_kl(mean,std,reference_mean,reference_std,mask=None):
    """Exact KL per complete action chunk, averaged over valid chunks."""
    mean=mean.float();std=std.float()
    reference_mean=reference_mean.detach().float();reference_std=reference_std.detach().float()
    if not(mean.shape==std.shape==reference_mean.shape==reference_std.shape):
        raise ValueError("transition_parameter_shapes_differ")
    selected=_expanded_mask(mean,mask)
    mean=torch.where(selected,mean,0);reference_mean=torch.where(selected,reference_mean,0)
    std=torch.where(selected,std,1);reference_std=torch.where(selected,reference_std,1)
    if bool((std<=0).any())or bool((reference_std<=0).any()):
        raise ValueError("KL_requires_positive_std")
    terms=(torch.log(reference_std/std)+(std.square()+(mean-reference_mean).square())/(2*reference_std.square())-.5).clamp_min(0)
    per_chunk=terms.reshape(terms.shape[0],-1).sum(-1)
    valid=selected.reshape(selected.shape[0],-1).any(-1)
    total,count=_masked_sum_count(per_chunk,valid)
    return total/count.clamp_min(1),count

def rollout_kl_sum_count(current,previous,mask):
    """k3 KL on the same complete-chunk density used by PPO."""
    delta=current.float()-previous.detach().float()
    selected=_expanded_mask(delta,mask);delta=torch.where(selected,delta,0)
    joint=delta.reshape(delta.shape[0],-1).sum(-1)
    valid=selected.reshape(selected.shape[0],-1).any(-1)
    if not torch.isfinite(joint[valid]).all():raise FloatingPointError("nonfinite_joint_log_ratio")
    if bool((joint[valid].abs()>40).any()):raise FloatingPointError("joint_log_ratio_exceeds_40")
    return _masked_sum_count(torch.expm1(joint)-joint,valid)

def init_reference(actor):
    from rlinf.models import get_model
    if float(actor.cfg.algorithm.kl_beta)<=0:raise ValueError("positive_SFT_KL_required")
    devices=[torch.cuda.current_device()]if torch.cuda.is_available()else[]
    with torch.random.fork_rng(devices=devices),torch.no_grad():
        reference=get_model(actor.cfg.actor.model)
    reference.requires_grad_(False);reference.eval();reference.to(actor.device)
    if reference.config.noise_method!="flow_sde"or reference.config.joint_logprob:
        raise ValueError("validated_KL_scope_requires_nonjoint_flow_sde")
    if reference.config.safe_get_logprob:raise ValueError("KL_guard_requires_normalized_Gaussian_logprobs")
    actor._sft_reference_model=reference
    folder=Path(os.environ["JOINT_RL_RUN"])/"runtime_checks";folder.mkdir(parents=True,exist_ok=True)
    receipt={"version":VERSION,"rank":actor._rank,"pid":os.getpid(),
             "critic_warmup_rounds":int(actor.cfg.algorithm.critic_warmup_rounds),
             "kl_beta":float(actor.cfg.algorithm.kl_beta),
             "target_rollout_kl":float(actor.cfg.algorithm.target_rollout_kl),
             "reference_checkpoint":str(actor.cfg.actor.model.model_path),
             "reference_trainable_parameters":sum(p.numel()for p in reference.parameters()if p.requires_grad),
             "reference_scope":"conditional stochastic denoising Gaussian; sum coordinates per chunk, mean valid chunks",
             "actor_module":str(__import__("inspect").getfile(type(actor))),
             "reference_module":str(__import__("inspect").getfile(type(reference)))}
    (folder/("actor_rank_"+str(actor._rank)+".json")).write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"event":"warmkl_reference_ready",**receipt}),flush=True)
def begin_round(actor):
    actor._stability_moments=[]
    actor._stability_old_kl=torch.zeros(2,device=actor.device,dtype=torch.float64)
    actor._stability_sft_kl=torch.zeros(2,device=actor.device,dtype=torch.float64)
    actor._stability_batch_old_kl=torch.zeros(2,device=actor.device,dtype=torch.float64)
    actor._stability_actor_stopped=False
    actor._stability_optimizer_skips=0
def extra_loss(actor,micro_batch,output,loss,metrics_data,loss_kwargs):
    warm=warmup_active(actor)
    actor._stability_moments.append(prediction_moments(
        output["values"],loss_kwargs["returns"],loss_kwargs["prev_values"],
        loss_kwargs["value_clip"],loss_kwargs.get("loss_mask")))
    mask=loss_kwargs.get("loss_mask")
    kl_sum,count=rollout_kl_sum_count(output["logprobs"],micro_batch["prev_logprobs"],mask)
    sample_kl=torch.stack((kl_sum.detach().double(),count.double()))
    actor._stability_old_kl+=sample_kl
    actor._stability_batch_old_kl+=sample_kl
    metrics_data["actor/critic_only_warmup"]=float(warm)
    metrics_data["actor/effective_sft_kl_beta"]=0. if warm else float(actor.cfg.algorithm.kl_beta)
    if not warm:
        with torch.no_grad(),actor.amp_context:
            reference=actor._sft_reference_model(
                forward_inputs=micro_batch["forward_inputs"],compute_logprobs=True,
                compute_entropy=False,compute_values=False,compute_transition_stats=True,use_cache=False)
        kl,n=gaussian_transition_kl(output["transition_mean"],output["transition_std"],
            reference["transition_mean"],reference["transition_std"],mask)
        actor._stability_sft_kl+=torch.stack(((kl.detach()*n).double(),n.double()))
        loss=loss+float(actor.cfg.algorithm.kl_beta)*kl
    return loss
def before_optimizer(actor):
    sums=_distributed_sum(actor._stability_batch_old_kl.clone())
    actor._stability_batch_old_kl.zero_()
    old_kl=float((sums[0]/sums[1].clamp_min(1)).item())
    if not warmup_active(actor)and old_kl>float(actor.cfg.algorithm.target_rollout_kl):
        actor._stability_actor_stopped=True
    if warmup_active(actor)or actor._stability_actor_stopped:
        kept=0
        for name,param in actor.model.named_parameters():
            if"value_head."not in name:param.grad=None
            elif param.grad is not None:kept+=1
        total_kept=_distributed_sum(torch.tensor(float(kept),device=actor.device))
        if total_kept.item()==0:raise RuntimeError("critic_only_update_has_no_value_head_gradients")
        if any(p.grad is not None for n,p in actor.model.named_parameters()if"value_head."not in n):
            raise RuntimeError("actor_gradient_survived_critic_only_guard")
        actor._stability_optimizer_skips+=1
def finish_round(actor,metrics):
    if not actor._stability_moments:raise RuntimeError("no_actual_critic_predictions_collected")
    moments=_distributed_sum(torch.stack(actor._stability_moments).sum(0))
    metrics.update(metrics_from_moments(moments.cpu().tolist()))
    old=_distributed_sum(actor._stability_old_kl.clone());ref=_distributed_sum(actor._stability_sft_kl.clone())
    metrics["actor/rollout_k3_kl_per_chunk"]=float((old[0]/old[1].clamp_min(1)).item())
    metrics["actor/sft_transition_kl_per_chunk"]=float((ref[0]/ref[1].clamp_min(1)).item())
    metrics["actor/kl_guard_triggered"]=float(actor._stability_actor_stopped)
    metrics["actor/critic_only_optimizer_steps"]=float(actor._stability_optimizer_skips)
    metrics["actor/critic_only_warmup"]=float(warmup_active(actor))
    metrics["actor/effective_sft_kl_beta"]=0. if warmup_active(actor)else float(actor.cfg.algorithm.kl_beta)
    metrics["actor/old_logprobs_recomputed"]=float(getattr(actor,"_stability_old_logprobs_recomputed",False))
    actor._stability_moments.clear()
    return metrics

def cache_actor_old_logprobs(actor):
    """Cache once before any policy updates, using exact training microbatches."""
    previous=actor.rollout_batch["prev_logprobs"]
    total=previous.shape[0];micro=int(actor.cfg.actor.micro_batch_size)
    if total%micro:raise ValueError("rollout_must_have_complete_training_microbatches")
    inputs=actor.rollout_batch["forward_inputs"]
    def part(value,start,end):
        if torch.is_tensor(value):return value[start:end].to(actor.device)
        if isinstance(value,dict):return{k:part(v,start,end)for k,v in value.items()}
        if isinstance(value,list):return value[start:end]
        return value
    for start in range(0,total,micro):
        with torch.enable_grad(),actor.amp_context:
            result=actor.model(forward_inputs=part(inputs,start,start+micro),
                compute_logprobs=True,compute_entropy=actor.cfg.algorithm.entropy_bonus>0,
                compute_values=actor.cfg.algorithm.adv_type=="gae",
                compute_transition_stats=True,use_cache=False)
            old=result["logprobs"].detach().to(device=previous.device,dtype=previous.dtype)
        if not torch.isfinite(old).all():raise FloatingPointError("nonfinite_recomputed_old_logprobs")
        previous[start:start+micro].copy_(old)
        del result,old
    actor._stability_old_logprobs_recomputed=True
