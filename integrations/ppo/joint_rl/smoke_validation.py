"""Reject smoke runs whose required PPO losses or gradients are not finite."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path


REQUIRED_TRAINING_TAGS = (
    "train/actor/policy_loss",
    "train/actor/total_loss",
    "train/actor/grad_norm",
    "train/critic/value_loss",
)


def validate_training_scalars(records: dict[str, list[float]]) -> dict:
    """Validate every recorded required loss/gradient scalar, including earlier steps."""
    problems = []
    counts = {}
    latest = {}
    for tag in REQUIRED_TRAINING_TAGS:
        if tag not in records:
            problems.append(f"missing required scalar {tag}")
            continue
        values = records[tag]
        if not values:
            problems.append(f"empty required scalar {tag}")
            continue
        counts[tag] = len(values)
        finite_values = []
        for index, value in enumerate(values):
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):
                problems.append(f"invalid scalar {tag}[{index}]={value!r}")
                continue
            if not math.isfinite(numeric):
                problems.append(f"nonfinite scalar {tag}[{index}]={numeric!r}")
                continue
            finite_values.append(numeric)
        if len(finite_values) == len(values):
            latest[tag] = finite_values[-1]
    if problems:
        raise RuntimeError(
            "smoke_numerical_validation_failed: " + "; ".join(problems)
            + ". Formal PPO must not proceed; inspect the smoke training.log and TensorBoard scalars."
        )
    return {
        "checked_tags": list(REQUIRED_TRAINING_TAGS),
        "sample_counts": counts,
        "latest_values": latest,
    }


def load_and_validate_smoke(folder: Path) -> dict:
    """Read actual smoke events and persist a pass/fail receipt before continuing."""
    folder = Path(folder)
    log_root = folder / "output" / "tensorboard"
    receipt_path = folder / "numerical_check.json"
    receipt = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "log_root": str(log_root),
        "checked_tags": list(REQUIRED_TRAINING_TAGS),
        "event_directories": [],
    }
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        event_dirs = sorted({path.parent for path in log_root.rglob("events.out.tfevents.*")})
        receipt["event_directories"] = [str(path) for path in event_dirs]
        if not event_dirs:
            raise RuntimeError(f"no TensorBoard event files found under {log_root}")
        records = {}
        for directory in event_dirs:
            accumulator = EventAccumulator(
                str(directory), size_guidance={"scalars": 0}, purge_orphaned_data=False,
            )
            accumulator.Reload()
            available = set(accumulator.Tags().get("scalars", []))
            for tag in REQUIRED_TRAINING_TAGS:
                if tag in available:
                    records.setdefault(tag, []).extend(float(event.value) for event in accumulator.Scalars(tag))
        result = validate_training_scalars(records)
        receipt.update(result, passed=True)
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
        raise RuntimeError(
            f"smoke_numerical_validation_failed; inspect {receipt_path}: {error}"
        ) from error
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return receipt
