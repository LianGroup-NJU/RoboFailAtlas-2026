from __future__ import annotations

import hashlib
import json
import math

SCHEMA = "qwen3-vl-rm-samples-v1"
SPLITS = ("train", "validation", "test")
A_VERSION = "rm-action-task-scoring-v2-20261023"
RECOVERY = ["recovered", "attempted_not_recovered", "not_recovered_as_of_checkpoint"]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def valid_score(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def targets(frame_count):
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("invalid_frame_count")
    return sorted({round(i * (frame_count - 1) / 15) for i in range(16)})


def causal_frames(available, current, mode, max_frames=16):
    if mode not in ("single", "history") or max_frames < 1:
        raise ValueError("invalid_input_protocol")
    frames = sorted(set(f for f in available if f <= current))
    if not frames or frames[-1] != current:
        raise ValueError("current_image_missing")
    if mode == "single":
        return [current]
    if max_frames == 1:
        return [current]
    if len(frames) > max_frames:
        frames = [frames[round(i * (len(frames) - 1) / (max_frames - 1))] for i in range(max_frames)]
    assert frames[-1] == current and max(frames) <= current
    return frames


def ontology(taxonomy):
    actions = sorted({a["code"] for m in taxonomy["modules"] for a in m["atoms"]})
    labels = {x["code"]: x["name"] for c in taxonomy["categories"] for x in c.get("labels", [])}
    types = sorted(c for c in labels if c.startswith(("FAIL.A", "FAIL.B", "FAIL.D", "FAIL.E")))
    results = sorted(c for c in labels if c.startswith("FAIL.C"))
    names = {**labels, **{a["code"]: a.get("label", a["code"]) for m in taxonomy["modules"] for a in m["atoms"]}}
    value = {"failure_actions": actions, "failure_types": types, "failure_results": results,
             "recovery": RECOVERY, "names": names,
             "taxonomy_revision": taxonomy.get("gemini_note_failure_extension", {}).get("taxonomy_revision", 22)}
    return {**value, "sha256": digest(value)}


def multilabel(codes, universe, negative=False, exhaustive=False):
    codes = set(codes)
    if not codes <= set(universe):
        raise ValueError("label_outside_frozen_ontology")
    # Unselected labels are unknown by default. A terminal, explicitly empty
    # event set on a successful candidate supplies negative examples.
    return {"values": [float(c in codes) for c in universe],
            "mask": [bool(negative or exhaustive or c in codes) for c in universe]}


def auxiliary_targets(result, current, terminal, outcome, labels, exhaustive=False):
    annotation = result["proposed_annotation"]
    events = annotation["events"]
    selected = []
    for event in events:
        # Coarse event intervals are not reliable onset labels. Be conservative:
        # use an event only once its explicit checkpoint and evidence are in view.
        checkpoint = event.get("checkpoint_frame")
        if checkpoint is None:
            checkpoint = event["end_frame"]
        analysis = result.get("analysis_assessment", {}).get("events", {}).get(event["id"], {})
        def frames(value):
            out = []
            if isinstance(value, dict):
                for k, v in value.items():
                    if k == "frames" and isinstance(v, list):
                        out += [f for f in v if type(f) is int]
                    else:
                        out += frames(v)
            elif isinstance(value, list):
                for x in value:
                    out += frames(x)
            return out
        cited = frames(analysis) + event.get("recovery_assessment", {}).get("frames", [])
        if checkpoint <= current and all(f <= current for f in cited):
            selected.append(event)
    codes = {c["code"] for event in selected for c in event["labels"]}
    negative = current == terminal and outcome == "success" and not events
    first_id = result.get("first_failure", {}).get("event_id")
    first = next((event for event in selected if event["id"] == first_id), None)
    by_id = {s["id"]: s for s in annotation["segments"]}
    actions = {by_id[s].get("atom_code") for s in first["failed_segment_ids"]} if first else set()
    actions.discard(None)
    values = {
        "failure_actions": multilabel(actions, labels["failure_actions"], negative, exhaustive and bool(actions)),
        "failure_types": multilabel(codes & set(labels["failure_types"]), labels["failure_types"], negative,
                                    exhaustive and bool(codes & set(labels["failure_types"]))),
        "failure_results": multilabel(codes & set(labels["failure_results"]), labels["failure_results"], negative,
                                      exhaustive and bool(codes & set(labels["failure_results"]))),
    }
    recovery = first.get("recovery_assessment", {}).get("status") if first else None
    values["recovery"] = {"value": labels["recovery"].index(recovery) if recovery in labels["recovery"] else 0,
                          "mask": recovery in labels["recovery"]}
    return values


def validate_sample(row):
    if row["schema"] != SCHEMA or row["split"] not in SPLITS:
        raise ValueError("invalid_sample_schema_or_split")
    if type(row["frame"]) is not int or not 0 <= row["frame"] < row["frame_count"]:
        raise ValueError("invalid_sample_frame")
    if not row["instruction"].strip() or not row["frames"] or row["frames"] != sorted(set(row["frames"])):
        raise ValueError("invalid_instruction_or_frames")
    if row["frames"][-1] != row["frame"]:
        raise ValueError("future_or_missing_current_frame")
    budget = row.get("history_max_frames", 16)
    if type(budget) is not int or budget < 1 or len(row["frames"]) > budget:
        raise ValueError("history_frame_budget_exceeded")
    if row["mode"] == "single" and row["frames"] != [row["frame"]]:
        raise ValueError("single_mode_must_have_one_frame")
    for key in ("A", "T"):
        target = row["targets"][key]
        if target["mask"] and not valid_score(target["value"]):
            raise ValueError("invalid_numeric_target")
    if row["targets"]["T"]["mask"] and row["frame"] != row["frame_count"] - 1:
        raise ValueError("environment_outcome_on_intermediate_frame")
    return row
