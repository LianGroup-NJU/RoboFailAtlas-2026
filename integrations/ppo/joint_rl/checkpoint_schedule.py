"""Preserve completed PPO updates before entering periodic evaluation."""

import importlib
from typing import Callable


def make_checkpoint_schedule(check_progress: Callable) -> Callable:
    """Keep RLinf's schedule and metric contract while saving before evaluation."""
    def _maybe_eval_and_checkpoint(self, step: int) -> dict:
        run_val, save_model, _ = check_progress(
            self.global_step,
            self.max_steps,
            self.cfg.runner.val_check_interval,
            self.cfg.runner.save_interval,
            1.0,
            run_time_exceeded=False,
        )

        if self.global_step in (10, 20, 50, 100):
            # RLinf's implementation waits for the actor's checkpoint handle.
            self._save_checkpoint()
            import os
            from pathlib import Path
            from common import complete_checkpoint, write
            folder=Path(os.environ['JOINT_RL_RUN'])
            matches=list((folder/'output').glob('*/checkpoints/global_step_'+str(self.global_step)))
            if len(matches)!=1 or not complete_checkpoint(matches[0]):
                raise RuntimeError('checkpoint_save_did_not_finish')
            write(folder/'checkpoint_ready'/(str(self.global_step)+'.json'),
                  {'checkpoint':str(matches[0]),'step':self.global_step,'complete':True})

        eval_metrics = {}
        if run_val:
            with self.timer("eval"):
                self.update_rollout_weights()
                eval_metrics = self.evaluate()
                eval_metrics = {f"eval/{key}": value for key, value in eval_metrics.items()}
                self.metric_logger.log(data=eval_metrics, step=step)

        return eval_metrics

    _maybe_eval_and_checkpoint._joint_rl_checkpoint_before_eval = True
    return _maybe_eval_and_checkpoint


def install() -> None:
    """Install the reordered driver method without changing the shared checkout."""
    module = importlib.import_module("rlinf.runners.embodied_runner")
    original = module.EmbodiedRunner._maybe_eval_and_checkpoint
    if getattr(original, "_joint_rl_checkpoint_before_eval", False):
        return
    replacement = make_checkpoint_schedule(module.check_progress)
    replacement.__wrapped__ = original
    module.EmbodiedRunner._maybe_eval_and_checkpoint = replacement
