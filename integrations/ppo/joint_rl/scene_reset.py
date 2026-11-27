"""Admit stable, expert-valid scenes before any policy action is executed."""

from dataclasses import dataclass
from joint_rl.policy_task_state import initialize_policy_task_state
from typing import Any, Callable


@dataclass
class SceneResetResult:
    task: Any
    episode_info: dict
    seed: int
    attempts: int
    rejections: list[dict]


def prepare_scene(
    make_task: Callable[[], Any], config: dict, seed: int, seed_stride: int,
    horizon: int, unstable_error: type[Exception], *, max_attempts: int = 100,
    on_event: Callable[[dict], None] | None = None, seed_upper_bound: int | None = None,
    on_release: Callable[[], None] | None = None,
) -> SceneResetResult:
    """Retry only rejected admission scenes; keep policy execution outside this loop."""
    if max_attempts < 1 or seed_stride < 1 or horizon < 1:
        raise ValueError("positive_scene_budget_stride_and_horizon_required")
    seed = int(seed)
    rejections = []

    def emit(event):
        if on_event is not None:
            on_event(event)

    def discard(task, candidate, stage, allow_incomplete=False):
        if task is None:
            return
        try:
            task.close_env()
        except AttributeError as error:
            # An initial setup can fail before the viewer/scene is fully assigned.
            if not allow_incomplete:
                raise
            emit({"event": "scene_cleanup_warning", "seed": candidate, "stage": stage,
                  "error": repr(error)})

    def try_candidate(candidate, attempt):
        # Candidate-local scope releases rejected-task tracebacks before the next
        # collection callback. No local may retain the expert while rollout loads.
        task = None
        stage = "expert_setup"
        try:
            task = make_task()
            task.setup_demo(now_ep_num=0, seed=candidate, is_test=True, **config)
            stage = "expert_validation"
            episode_info = task.play_once()
            valid = bool(task.plan_success and task.check_success())
            if not valid:
                rejected = {"event": "scene_rejected", "requested_seed": seed,
                            "seed": candidate, "attempt": attempt + 1, "stage": stage,
                            "reason": "expert_validation_failed"}
                discard(task, candidate, stage)
                task = None
                return None, None, rejected
            discard(task, candidate, stage)
            task = None
            if on_release is not None:
                on_release()
            # The expert run must never become the policy's initial scene.
            stage = "rollout_setup"
            task = make_task()
            task.setup_demo(now_ep_num=0, seed=candidate, is_test=True, **config)
            stage = "rollout_metadata"
            metadata = initialize_policy_task_state(task, episode_info)
            if metadata:
                emit({"event": "scene_task_metadata_initialized", "seed": candidate, **metadata})
            task.step_lim = horizon
            return task, episode_info, None
        except unstable_error as error:
            discard(task, candidate, stage, allow_incomplete=stage.endswith("setup"))
            task = None
            rejected = {"event": "scene_rejected", "requested_seed": seed,
                        "seed": candidate, "attempt": attempt + 1, "stage": stage,
                        "reason": type(error).__name__, "error": str(error)}
            return None, None, rejected
        except BaseException as error:
            # A missing grasp pose during expert admission rejects this candidate.
            # Never turn assertions during rollout setup or policy execution into
            # scene retries, and never suppress any other assertion message.
            if (stage == "expert_validation" and isinstance(error, AssertionError)
                    and str(error) == "target_pose cannot be None for move action."):
                discard(task, candidate, stage)
                task = None
                return None, None, {
                    "event": "scene_rejected", "requested_seed": seed,
                    "seed": candidate, "attempt": attempt + 1, "stage": stage,
                    "reason": "expert_grasp_target_unavailable", "error": str(error)}
            # Preserve the original programming/infrastructure error if cleanup fails too.
            try:
                discard(task, candidate, stage, allow_incomplete=True)
            except Exception as cleanup_error:
                emit({"event": "scene_cleanup_warning", "seed": candidate, "stage": stage,
                      "error": repr(cleanup_error)})
            raise

    for attempt in range(max_attempts):
        # The previous caught exception and candidate frame are now out of scope.
        if on_release is not None:
            on_release()
        candidate = seed + attempt * seed_stride
        if seed_upper_bound is not None and candidate >= seed_upper_bound:
            raise RuntimeError("scene_candidate_seed_namespace_exhausted")
        task, episode_info, rejected = try_candidate(candidate, attempt)
        if rejected is not None:
            rejections.append(rejected)
            emit(rejected)
            continue
        result = SceneResetResult(task, episode_info, candidate, attempt + 1, rejections)
        emit({"event": "scene_ready", "requested_seed": seed, "seed": candidate,
              "attempts": result.attempts, "rejected_candidates": len(rejections)})
        return result
    if on_release is not None:
        on_release()
    raise RuntimeError(f"no_expert_valid_scene_in_candidate_budget: seed={seed}, attempts={max_attempts}")
