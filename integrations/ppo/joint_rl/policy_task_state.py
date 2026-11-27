"""Initialize only task metadata needed by a fresh policy scene."""
import importlib
import math

def initialize_policy_task_state(task, episode_info=None):
    if type(task).__name__ != "place_object_scale":
        return {}
    # Match the official play_once rule using the fresh initial scene.
    # Do not transfer expert objects, poses, success flags, or action history.
    x = float(task.object.get_pose().p[0])
    if not math.isfinite(x):
        raise ValueError("nonfinite_initial_object_position")
    side = "right" if x > 0 else "left"
    info = (episode_info or {}).get("info", {})
    expert_side = info.get("{a}")
    if expert_side is not None and str(expert_side) != side:
        raise ValueError("fresh_scene_arm_disagrees_with_admitted_expert")
    module = importlib.import_module(type(task).__module__)
    task.arm_tag = module.ArmTag(side)
    return {"fields": ["arm_tag"], "arm": side,
            "source": "fresh_initial_object_position"}
