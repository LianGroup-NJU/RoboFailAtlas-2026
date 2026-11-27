"""RLinf-compatible vector adapter for the checkpoint's actual Franka simulator."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

from .reward import RewardClient, at_reward, terminal_t_reward
from .structured_reward import StructuredRewardTracker, profile_hash, structured_rm_url


from .resilient_sim_process import SimProcess


class FrankaRewardEnv(RoboTwinEnv):
    """Preserve RLinf's reset/metric contract; score real observations before reset."""

    def _uses_structured(self):
        return not self.cfg.is_eval and getattr(self.cfg, "reward_source", "joint") == "joint_structured"

    def _uses_rm(self):
        source = getattr(self.cfg, "reward_source", "joint")
        if source not in ("joint", "joint_structured", "robometer", "environment", "full_T"):
            raise ValueError("unsupported_reward_source:" + source)
        return not self.cfg.is_eval and source != "environment"

    def _init_reset_state_ids(self):
        self.slot = self.seed_offset * self.num_envs
        initial = 0 if self.cfg.is_eval else int(os.environ.get("JOINT_RL_INITIAL_SCENE_SERIAL", "0"))
        if initial < 0 or self.base_seed + self.slot + self.num_envs + 100000 * initial >= 2_900_000_000 and not self.cfg.is_eval:
            raise ValueError("invalid_initial_training_scene_serial")
        self.serial = np.full(self.num_envs, initial, dtype=np.int64)
        self.reset_state_ids = torch.tensor([self.base_seed + self.slot + i + 100000 * initial for i in range(self.num_envs)])

    def update_reset_state_ids(self, env_idx=None):
        # Evaluation repeats fixed candidate streams; training uses new streams.
        if self.cfg.is_eval:
            return
        ids = list(range(self.num_envs)) if env_idx is None else [int(i) for i in env_idx]
        for i in ids:
            self.serial[i] += 1
            seed = self.base_seed + self.slot + i + 100000 * int(self.serial[i])
            if seed >= 2_900_000_000:
                raise RuntimeError("training_scene_namespace_exhausted")
            self.reset_state_ids[i] = seed

    def _init_env(self):
        work = Path(os.environ["JOINT_RL_RUN"]) / "simulators" / ("eval" if self.cfg.is_eval else "train") / str(self.seed_offset)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if not visible or not visible[0]:
            raise RuntimeError("CUDA_visible_devices_required_for_renderer_binding")
        gpu = visible[torch.cuda.current_device()]
        self.simulators = [SimProcess(work / str(i), self.cfg.franka_repo, self.cfg.sim_python, gpu) for i in range(self.num_envs)]
        self.cached = [None] * self.num_envs
        self.histories = [[] for _ in range(self.num_envs)]
        self.previous_a = [0.0] * self.num_envs
        self.finished = np.zeros(self.num_envs, dtype=bool)
        self.structured_trackers = None
        if self._uses_structured():
            profile = json.loads(Path(os.environ["JOINT_STRUCTURED_PROFILE"]).read_text())
            if profile_hash(profile) != os.environ["JOINT_STRUCTURED_PROFILE_SHA256"]:
                raise ValueError("structured_reward_profile_changed")
            self.structured_trackers = [StructuredRewardTracker(profile) for _ in range(self.num_envs)]
            folder = Path(os.environ["JOINT_RL_RUN"])/"structured_rewards"
            folder.mkdir(exist_ok=True)
            self.structured_reward_log = folder/("rank_"+str(self.seed_offset)+".jsonl")
        url = self.cfg.rm_url
        if self._uses_structured():
            url = structured_rm_url(url,self.seed_offset,os.environ.get('STRUCTURED_RM_URLS'))
        if getattr(self.cfg, 'reward_source', '') == 'full_T':
            urls = json.loads(os.environ['FULL_T_RM_URLS'])
            url = urls[int(self.seed_offset) % len(urls)]
            folder = Path(os.environ['JOINT_RL_RUN'])/'full_T_rewards'
            folder.mkdir(exist_ok=True)
            self.full_t_reward_log = folder/('rank_'+str(self.seed_offset)+'.jsonl')
        self.client = RewardClient(url, self.cfg.rm_checkpoint_sha256) if self._uses_rm() else None

    def _observations(self):
        rows = [row["obs"] for row in self.cached]
        return {
            "main_images": torch.from_numpy(np.stack([r["images"]["head"] for r in rows])),
            "wrist_images": torch.from_numpy(np.stack([np.stack([r["images"]["left"], r["images"]["right"]]) for r in rows])),
            "extra_view_images": None,
            "states": torch.from_numpy(np.stack([r["state"] for r in rows])),
            "task_descriptions": [r["instruction"] for r in rows],
        }

    def reset(self, env_idx=None, env_seeds=None):
        ids = list(range(self.num_envs)) if env_idx is None else [int(i) for i in env_idx]
        self._is_start = False
        for i in ids:
            seed = int(self.reset_state_ids[i]) if env_seeds is None else int(env_seeds[i])
            self.simulators[i].send({"operation": "reset", "task": self.task_name, "seed": seed,
                                     "seed_stride": 1000, "horizon": self.cfg.max_episode_steps,
                                     "seed_upper_bound": None if self.cfg.is_eval else 2_900_000_000})
        for i in ids:
            self.cached[i] = self.simulators[i].receive()
            self.histories[i] = self.cached[i]["head_frames"]
            self.finished[i] = False
        self._reset_metrics(env_idx)
        if self._uses_rm() and getattr(self.cfg, "reward_source", "") != "full_T":
            scores = self.client.score([self.histories[i] for i in ids], [self.cached[i]["obs"]["instruction"] for i in ids])
            for i, score in zip(ids, scores, strict=True):
                self.previous_a[i] = score["A"]
                if self.structured_trackers is not None:
                    self.structured_trackers[i].reset(score)
        return self._observations(), {}

    def chunk_step(self, chunk_actions):
        actions = np.asarray(chunk_actions, dtype=np.float32)
        batch, chunk, dimensions = actions.shape
        if batch != self.num_envs or dimensions != 16:
            raise ValueError("expected_num_envs_by_chunk_by_joint16")
        active = [i for i in range(self.num_envs) if not self.finished[i]]
        for i in active:
            self.simulators[i].send({"operation": "step", "actions": actions[i]})
        for i in active:
            self.cached[i] = self.simulators[i].receive()
            self.histories[i].extend(self.cached[i]["head_frames"])
        values = torch.zeros(self.num_envs, device=self.device)
        terminal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros_like(terminal)
        success = torch.tensor([bool(row["success"]) for row in self.cached], device=self.device)
        full_t = getattr(self.cfg, 'reward_source', '') == 'full_T'
        score_ids = [i for i in active if self.cached[i]['done']] if full_t else active
        scores = None
        if self._uses_rm() and active:
            scores = [{'A': 0.0, 'T': 0.0} for i in active]
            if score_ids:
                predicted = self.client.score([self.histories[i] for i in score_ids],
                    [self.cached[i]['obs']['instruction'] for i in score_ids])
                positions = {index: position for position,index in enumerate(active)}
                for index, score in zip(score_ids, predicted, strict=True):
                    scores[positions[index]] = score
        for j, i in enumerate(active):
            row = self.cached[i]
            self._elapsed_steps[i] = int(row["steps"])
            terminal[i] = row["done"] if self.cfg.get("task_horizon_is_terminal", False) else row["success"]
            truncated[i] = row["done"] and not bool(terminal[i])
            self.finished[i] = bool(row["done"])
            if not self.cfg.is_eval and getattr(self.cfg, "reward_source", "joint") == "environment":
                values[i] = float(bool(row["done"]) and bool(row["success"]))
            elif scores is not None:
                score = scores[j]
                if full_t:
                    values[i] = terminal_t_reward(score['T'], bool(row['done']))
                    with self.full_t_reward_log.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps({'slot': i, 'control_step': int(row['steps']),
                            'terminal': bool(row['done']), 'scored': i in score_ids,
                            'T': float(score['T']) if i in score_ids else None,
                            'reward': float(values[i]), 'head_frames_observed': len(self.histories[i]),
                            'A_used': False, 'structured_used': False}, allow_nan=False)+'\n')
                elif self.structured_trackers is None:
                    values[i] = at_reward(self.previous_a[i], score["A"], score["T"], bool(row["done"]))
                else:
                    components = self.structured_trackers[i].step(score, terminal=bool(row["done"]))
                    values[i] = components["reward_total"]
                    audit = {"slot": i, "control_step": int(row["steps"]), "terminal": bool(row["done"]),
                             "environment_success_for_evaluation_only": bool(row["success"]), **components}
                    with self.structured_reward_log.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(audit, allow_nan=False)+"\n")
                self.previous_a[i] = score["A"]
        infos = self._record_metrics(values, {"success": success})
        final_obs = self._observations()
        dones = terminal | truncated
        obs = final_obs
        if self.auto_reset and bool(dones.any()):
            final_info = infos.copy()
            ids = torch.where(dones)[0].tolist()
            self.update_reset_state_ids(ids)
            obs, infos = self.reset(ids)
            infos.update(final_observation=final_obs, final_info=final_info,
                         _final_info=dones, _final_observation=dones, _elapsed_steps=dones)
        rewards = torch.zeros((batch, chunk), device=self.device)
        rewards[:, -1] = values
        terms = torch.zeros((batch, chunk), dtype=torch.bool)
        truncs = torch.zeros_like(terms)
        terms[:, -1] = terminal.cpu()
        truncs[:, -1] = truncated.cpu()
        return [obs], rewards, terms, truncs, [infos]

    def close(self, clear_cache=True):
        for process in getattr(self, "simulators", []):
            process.close()

    def sample_action_space(self):
        return np.zeros((self.num_envs, 50, 16), dtype=np.float32)

