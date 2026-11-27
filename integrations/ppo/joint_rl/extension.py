"""Use RLinf's EXT_MODULE hook without modifying the shared training checkout."""
from __future__ import annotations

import functools

_REGISTERED = False


def validate_initial_keys(expected: dict, state: dict) -> list[str]:
    """Permit initialization of the new PPO critic only; require every SFT weight."""
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if unexpected or any(not name.startswith("value_head.") for name in missing):
        raise ValueError(f"SFT_checkpoint_key_mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}")
    for name, value in state.items():
        if expected[name].shape != value.shape:
            raise ValueError("SFT_checkpoint_shape_mismatch:" + name)
    return sorted(missing)


def register():
    global _REGISTERED
    if _REGISTERED:
        return
    import torch
    import rlinf.models.embodiment.openpi.checkpoint_loading as loading
    import rlinf.envs.robotwin.robotwin_env as robotwin
    from .franka_env import FrankaRewardEnv

    original_load = loading.load_checkpoint_weights

    @functools.wraps(original_load)
    def load_weights(model, checkpoint_dir, *, strict=False):
        from pathlib import Path
        if getattr(model.config, "config_name", "") != "pi05_dualfranka_robotwin_head":
            return original_load(model, checkpoint_dir, strict=strict)
        path = Path(checkpoint_dir) / "actor/model_state_dict/full_weights.pt"
        if not path.is_file():
            return original_load(model, checkpoint_dir, strict=True)
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        initialized = validate_initial_keys(model.state_dict(), state)
        loading._validate_alias_values(loading._shared_alias_groups(model), state.keys(), state.__getitem__)
        model.load_state_dict(state, strict=not initialized)
        model.logger.info("Loaded all SFT parameters; newly initialized PPO critic parameters: %s", initialized)

    loading.load_checkpoint_weights = load_weights
    robotwin.RoboTwinEnv = FrankaRewardEnv
    from pathlib import Path
    import inspect
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    expected = Path(__import__("os").environ["FAILATLAS_RLINF_RUNTIME"]) / "rlinf/workers/actor/fsdp_actor_worker.py"
    if Path(inspect.getfile(EmbodiedFSDPActor)).resolve() != expected.resolve():
        raise RuntimeError("private_Embodied_PPO_overlay_not_loaded")
    _REGISTERED = True
