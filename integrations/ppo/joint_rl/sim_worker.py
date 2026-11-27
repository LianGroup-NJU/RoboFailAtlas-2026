"""Run one official Franka environment in its existing Python environment."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from multiprocessing.connection import Connection

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    connection = Connection(args.fd)
    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / "XPolicyLab"), str(repo / "XPolicyLab/policy/Pi05_Franka_40000")]
    from sapien_gpu_binding import gpu_inventory, install_binding

    token = args.gpu
    physical = int(token) if token.isdigit() else next(row["gpu_index"] for row in gpu_inventory() if row["gpu_uuid"] == token)
    binding = install_binding(physical)
    spec = importlib.util.spec_from_file_location("franka_official_evaluator", repo / "scripts/eval_policy_xpolicylab.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    task = None
    config = None
    text = None
    from joint_rl.reward import encode_frame
    from joint_rl.scene_reset import prepare_scene
    from envs.utils.create_actor import UnStableError
    from sapien.render import clear_cache as clear_render_cache

    def release_scene_memory():
        import torch
        gc.collect()
        clear_render_cache()
        torch.cuda.empty_cache()
        print(json.dumps({"event": "scene_memory_released", "pid": os.getpid(),
                          "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
                          "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 2)}), flush=True)

    def observation():
        raw = task.get_obs()
        xpl = evaluator.robotwin_obs_to_xpolicylab(raw, instruction=text, task_env=task)
        cameras = xpl["vision"]
        state = xpl["state"]
        values = np.concatenate([np.asarray(state[key]).reshape(-1) for key in
                                 ("left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state")]).astype(np.float32)
        pictures = {target: np.asarray(cameras[source]["color"]) for target, source in
                    (("head", "cam_head"), ("left", "cam_left_wrist"), ("right", "cam_right_wrist"))}
        if values.shape != (16,) or not np.isfinite(values).all():
            raise ValueError("nonfinite_or_invalid_joint16_state")
        if any(p.dtype != np.uint8 or p.ndim != 3 or p.shape[-1] != 3 for p in pictures.values()):
            raise ValueError("invalid_native_RGB_observation")
        return {"state": values, "images": pictures, "instruction": text}

    try:
        connection.send({"ready": True, "binding": binding})
        while True:
            request = connection.recv()
            operation = request["operation"]
            if operation == "close":
                break
            if operation == "reset":
                if task is not None:
                    task.close_env()
                    task = None
                episode_info, raw, obs = None, None, None
                release_scene_memory()
                name = request["task"]
                config, _ = evaluator.load_task_args({"task_name": name, "task_config": "baseline_franka_clean",
                                                     "policy_name": "Pi05_Franka_40000", "ckpt_name": "joint_rm_ppo"})
                if config["embodiment"] != ["franka-panda", "franka-panda", 0.8]:
                    raise ValueError("expected_frozen_dual_Franka_config")
                config.update(eval_mode=True, render_freq=0, save_data=False, eval_video_log=False,
                              save_path=str(args.work / "data"))
                result = prepare_scene(
                    lambda: evaluator.class_decorator(name), config,
                    int(request["seed"]), int(request["seed_stride"]), int(request["horizon"]),
                    UnStableError, max_attempts=100, seed_upper_bound=request.get("seed_upper_bound"),
                    on_event=lambda event: print(json.dumps({"task": name, **event}), flush=True),
                    on_release=release_scene_memory,
                )
                task, episode_info, seed = result.task, result.episode_info, result.seed
                scene_attempts, scene_rejections = result.attempts, result.rejections
                result = None
                text = evaluator.build_instruction(config, episode_info, "seen", 100)
                if not isinstance(text, str) or not text.strip() or text == name:
                    raise ValueError("natural_language_task_instruction_required")
                task.set_instruction(instruction=text)
                obs = observation()
                connection.send({"obs": obs, "seed": seed, "requested_seed": int(request["seed"]),
                                 "scene_attempts": scene_attempts, "scene_rejections": scene_rejections,
                                 "head_frames": [encode_frame(obs["images"]["head"])], "success": False})
                print(json.dumps({"event": "reset", "seed": seed, "instruction": text, "expert_valid": True}), flush=True)
            elif operation == "step":
                if task is None:
                    raise RuntimeError("reset_required")
                frames = []
                actions = np.asarray(request["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[-1] != 16 or not np.isfinite(actions).all():
                    raise ValueError("invalid_joint16_actions")
                executed = 0
                for action in actions:
                    if evaluator.is_episode_end(task):
                        break
                    raw = task.get_obs()
                    flat, kind = evaluator.xpolicylab_action_to_robotwin(action, action_type="joint", current_observation=raw)
                    task.take_action(flat, action_type=kind)
                    obs = observation()
                    frames.append(encode_frame(obs["images"]["head"]))
                    executed += 1
                    if evaluator.is_episode_end(task):
                        break
                obs = observation()
                success = bool(task.eval_success)
                connection.send({"obs": obs, "head_frames": frames, "executed": executed, "success": success,
                                 "done": evaluator.is_episode_end(task), "steps": task.take_action_cnt})
            else:
                raise ValueError("unknown_sim_operation")
    except EOFError:
        pass
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            connection.send({"error": error})
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        if task is not None:
            task.close_env()
            task = None
        release_scene_memory()
        connection.close()


if __name__ == "__main__":
    main()
