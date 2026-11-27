"""Episode-local, bounded shaping from a frozen joint reward model.

Failure type/result heads describe accumulated observed failures, so persistent
labels must not become repeated per-step costs. These engineering defaults are
an experimental reward profile, not calibrated probabilities or causal labels.
The failure-action head gates type costs; it never makes an action name bad.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real
from typing import Any


SCHEMA = "structured-rm-reward-v1"
RECOVERY_CLASSES = frozenset(
    ("recovered", "attempted_not_recovered", "not_recovered_as_of_checkpoint")
)
PROBABILITY_HEADS = ("failure_actions", "failure_types", "failure_results", "recovery")


def default_profile() -> dict[str, Any]:
    """Return fresh defaults for the bounded structured-reward experiment.

    ``type_weights`` and ``result_weights`` map supported label codes to
    nonnegative weights. ``None`` permits all returned codes with weight one;
    an empty mapping disables that head. Returned payloads must still include
    all probability heads when a head is disabled.
    """
    return {
        "schema": SCHEMA,
        "high_threshold": 0.8,
        "low_threshold": 0.5,
        "confirmation_steps": 2,
        "type_label_weight": 0.025,
        "type_step_cap": 0.05,
        "type_episode_cap": 0.10,
        "result_terminal_cap": 0.05,
        "max_recovery_refund": 0.05,
        "recovery_refund_fraction": 0.5,
        "terminal_success_threshold": 0.8,
        "type_weights": None,
        "result_weights": None,
        "action_supported_codes": None,
    }


def _number(value: Any, label: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label}_must_be_a_real_number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (maximum is not None and result > maximum):
        raise ValueError(f"{label}_nonfinite_or_out_of_range")
    return result


def _weight_map(value: Any, label: str) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}_must_be_a_code_to_weight_mapping")
    out = {}
    for code, weight in value.items():
        if not isinstance(code, str) or not code:
            raise ValueError(f"{label}_contains_invalid_code")
        out[code] = _number(weight, f"{label}.{code}")
    return out


def normalize_profile(profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate a profile and fill unspecified fields from the fixed defaults."""
    out = default_profile()
    if profile is not None:
        if not isinstance(profile, Mapping):
            raise ValueError("reward_profile_must_be_a_mapping")
        unknown = set(profile) - set(out)
        if unknown:
            raise ValueError("unknown_reward_profile_fields:" + ",".join(sorted(unknown)))
        out.update(profile)
    if out["schema"] != SCHEMA:
        raise ValueError("unsupported_structured_reward_schema")
    for key in ("high_threshold", "low_threshold", "terminal_success_threshold"):
        out[key] = _number(out[key], key, maximum=1.0)
    if not out["low_threshold"] < out["high_threshold"]:
        raise ValueError("low_threshold_must_be_lower_than_high_threshold")
    steps = out["confirmation_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("confirmation_steps_must_be_a_positive_integer")
    for key in (
        "type_label_weight", "type_step_cap", "type_episode_cap",
        "result_terminal_cap", "max_recovery_refund",
    ):
        out[key] = _number(out[key], key)
    # This bound is an invariant, not a tunable invitation to reward failure.
    out["recovery_refund_fraction"] = _number(
        out["recovery_refund_fraction"], "recovery_refund_fraction", maximum=0.5
    )
    out["type_weights"] = _weight_map(out["type_weights"], "type_weights")
    out["result_weights"] = _weight_map(out["result_weights"], "result_weights")
    action_codes = out["action_supported_codes"]
    if action_codes is not None:
        if not isinstance(action_codes, (list, tuple)) or any(
            not isinstance(code, str) or not code for code in action_codes
        ):
            raise ValueError("action_supported_codes_must_be_a_code_list_or_null")
        if len(set(action_codes)) != len(action_codes):
            raise ValueError("action_supported_codes_contains_duplicates")
        out["action_supported_codes"] = sorted(action_codes)
    return out


def profile_hash(profile: Mapping[str, Any] | None = None) -> str:
    """Hash the complete normalized reward definition for checkpoint lineage."""
    data = json.dumps(
        normalize_profile(profile), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _probability_map(score: Mapping[str, Any], head: str) -> dict[str, float]:
    canonical = head + "_probabilities"
    if canonical in score:
        value = score[canonical]
    elif head in score and isinstance(score[head], Mapping):
        value = score[head]
    else:
        raise ValueError("missing_probability_head:" + head)
    if not isinstance(value, Mapping) or not value:
        raise ValueError("probability_head_must_be_a_nonempty_mapping:" + head)
    result = {}
    for code, probability in value.items():
        if not isinstance(code, str) or not code:
            raise ValueError("invalid_probability_code:" + head)
        result[code] = _number(probability, f"{head}.{code}", maximum=1.0)
    if head == "recovery":
        if set(result) != RECOVERY_CLASSES:
            raise ValueError("recovery_probability_classes_mismatch")
        if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError("recovery_probabilities_must_sum_to_one")
    return result


def validate_score(score: Mapping[str, Any]) -> dict[str, Any]:
    """Validate A/T and every exported probability, including unsupported codes.

    The canonical API uses ``<head>_probabilities`` for probability mappings.
    For callers without separate display labels, the short head names can also
    contain mappings. Short list-valued display labels are ignored when the
    canonical probability field is present.
    """
    if not isinstance(score, Mapping):
        raise ValueError("RM_score_must_be_a_mapping")
    if "A" not in score or "T" not in score:
        raise ValueError("missing_A_or_T_score")
    result = {key: _number(score[key], key, maximum=1.0) for key in ("A", "T")}
    result.update({head: _probability_map(score, head) for head in PROBABILITY_HEADS})
    return result


class StructuredRewardTracker:
    """Track one environment's reward state until its next reset.

    Use one tracker per environment slot. A type is charged at most once per
    episode, even if it disappears and returns. Episode-level types therefore
    cannot identify a second instance of the same failure category.

    Recovery is a conditional terminal refund, not a positive per-step bonus.
    Its total refund is at most half of previously paid type costs, ensuring
    that the episode's structured additions cannot create net positive reward.
    """

    def __init__(self, profile: Mapping[str, Any] | None = None):
        self.profile = normalize_profile(profile)
        self.profile_sha256 = profile_hash(self.profile)
        self._initialized = False
        self._terminal = False

    def _supported(
        self, probabilities: Mapping[str, float], field: str
    ) -> dict[str, float]:
        weights = self.profile[field]
        if weights is None:
            return {code: 1.0 for code in probabilities}
        missing = set(weights) - set(probabilities)
        if missing:
            raise ValueError("supported_codes_missing_from_RM:" + ",".join(sorted(missing)))
        return {code: weight for code, weight in weights.items() if weight > 0}

    def reset(self, score: Mapping[str, Any]) -> None:
        """Initialize from the reset image without issuing shaping reward."""
        if profile_hash(self.profile) != self.profile_sha256:
            raise RuntimeError("reward_profile_changed_after_tracker_initialization")
        current = validate_score(score)
        type_weights = self._supported(current["failure_types"], "type_weights")
        result_weights = self._supported(current["failure_results"], "result_weights")
        action_codes = self.profile["action_supported_codes"]
        if action_codes is not None and not set(action_codes) <= set(current["failure_actions"]):
            raise ValueError("supported_action_codes_missing_from_RM")
        self._action_codes = (
            tuple(current["failure_actions"]) if action_codes is None else tuple(action_codes)
        )
        self._keys = {head: frozenset(current[head]) for head in PROBABILITY_HEADS}
        self._type_weights = type_weights
        self._result_weights = result_weights
        self._seen_types = {
            code for code in type_weights
            if current["failure_types"][code] >= self.profile["high_threshold"]
        }
        self._type_streaks = {code: 0 for code in type_weights}
        self._episode_type_cost = 0.0
        self._recovery_armed = False
        self._recovery_streak = 0
        self._recovery_confirmed = False
        self._previous_recovery = current["recovery"]["recovered"]
        self._previous_a = current["A"]
        self._initialized = True
        self._terminal = False

    def step(self, score: Mapping[str, Any], terminal: bool) -> dict[str, Any]:
        """Compute one chunk's decomposition and update episode-local state.

        Args:
            score: Current frozen-RM output, containing all probability heads.
            terminal: Whether this is the real episode endpoint.

        Returns:
            The scalar total and auditable components. Result probabilities are
            historical failure descriptors: they are used once only when the
            terminal RM T is below the profile's success threshold.
        """
        if not self._initialized:
            raise RuntimeError("structured_reward_requires_reset")
        if self._terminal:
            raise RuntimeError("structured_reward_episode_already_terminal")
        if profile_hash(self.profile) != self.profile_sha256:
            raise RuntimeError("reward_profile_changed_after_tracker_initialization")
        if not isinstance(terminal, bool):
            raise ValueError("terminal_must_be_boolean")
        current = validate_score(score)
        if any(frozenset(current[head]) != self._keys[head] for head in PROBABILITY_HEADS):
            raise ValueError("RM_probability_ontology_changed_during_episode")

        high = self.profile["high_threshold"]
        low = self.profile["low_threshold"]
        action_context = max(
            (current["failure_actions"][code] for code in self._action_codes), default=0.0
        )
        context_confirmed = action_context >= high
        new_types = []
        for code in self._type_weights:
            high_now = context_confirmed and current["failure_types"][code] >= high
            self._type_streaks[code] = self._type_streaks[code] + 1 if high_now else 0
            if code not in self._seen_types and self._type_streaks[code] >= self.profile["confirmation_steps"]:
                new_types.append(code)
        raw_type_cost = self.profile["type_label_weight"] * sum(
            current["failure_types"][code] * self._type_weights[code] for code in new_types
        )
        remaining = max(0.0, self.profile["type_episode_cap"] - self._episode_type_cost)
        type_cost = min(raw_type_cost, self.profile["type_step_cap"], remaining)
        previous_type_cost = self._episode_type_cost
        self._episode_type_cost += type_cost
        self._seen_types.update(new_types)

        recovered = current["recovery"]["recovered"]
        if previous_type_cost <= 0 < self._episode_type_cost:
            # A low value immediately before the first paid failure, or at that
            # step, can establish a real low-to-high recovery transition. An
            # always-high reset prediction cannot do so.
            self._recovery_armed = self._previous_recovery < low or recovered < low
            self._recovery_streak = 0
        if self._episode_type_cost > 0 and not self._recovery_confirmed:
            if recovered < low:
                self._recovery_armed = True
            if self._recovery_armed and recovered >= high:
                self._recovery_streak += 1
            else:
                self._recovery_streak = 0
            if self._recovery_streak >= self.profile["confirmation_steps"]:
                self._recovery_confirmed = True

        result_cost = 0.0
        recovery_refund = 0.0
        terminal_success = current["T"] if terminal else 0.0
        if terminal:
            if current["T"] < self.profile["terminal_success_threshold"]:
                # Gate by raw probability; severity weights change amount, not
                # confidence. Max pooling avoids multiplying correlated labels.
                result_signal = max(
                    (current["failure_results"][code] * weight
                     for code, weight in self._result_weights.items()
                     if current["failure_results"][code] >= high),
                    default=0.0,
                )
                result_cost = self.profile["result_terminal_cap"] * min(1.0, result_signal) * (1.0-current["T"])
            elif self._recovery_confirmed:
                recovery_refund = min(
                    self.profile["max_recovery_refund"],
                    self.profile["recovery_refund_fraction"] * self._episode_type_cost,
                ) * current["T"]
        progress = current["A"] - self._previous_a
        base = progress + terminal_success
        result = {
            "reward_total": base - type_cost - result_cost + recovery_refund,
            "base": base,
            "progress": progress,
            "terminal_success": terminal_success,
            "type_cost": type_cost,
            "result_cost": result_cost,
            "recovery_refund": recovery_refund,
            "episode_type_cost": self._episode_type_cost,
            "new_failure_types": sorted(new_types),
            "charged_failure_types": sorted(new_types) if type_cost > 0 else [],
            "failure_action_context": action_context,
            "failure_action_context_confirmed": context_confirmed,
            "recovery_confirmed": self._recovery_confirmed,
            "recovered_probability": recovered,
            "terminal": terminal,
            "A": current["A"],
            "T": current["T"],
        }
        if not math.isfinite(result["reward_total"]):
            raise ValueError("nonfinite_structured_reward")
        self._previous_a = current["A"]
        self._previous_recovery = recovered
        self._terminal = terminal
        return result

def structured_rm_url(default_url, rank, urls_json=None):
    """Route stateless scoring requests among identical frozen RM replicas."""
    if urls_json is None:
        return default_url
    urls = json.loads(urls_json)
    if not isinstance(urls,list) or not urls or any(not isinstance(u,str) or not u.startswith('http://127.0.0.1:') for u in urls):
        raise ValueError('invalid_local_structured_RM_replicas')
    return urls[int(rank) % len(urls)]
