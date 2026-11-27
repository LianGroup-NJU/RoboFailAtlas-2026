from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from .contracts import A_VERSION, SCHEMA, SPLITS, auxiliary_targets, canonical, causal_frames, digest, ontology, targets, valid_score, validate_sample


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(run, output, max_train_cases=0, max_other_cases=0, exhaustive=False, history_max_frames=16):
    if type(history_max_frames) is not int or history_max_frames < 1:
        raise ValueError("invalid_history_frame_budget")
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("snapshot_destination_exists_use_a_new_directory")
    config = json.loads((run / "run.json").read_text())
    if not config.get("full_reannotation"):
        raise ValueError("full_reannotation_round_required")
    for relative, expected in config["frozen_sha256"].items():
        if sha(run / relative) != expected:
            raise ValueError("frozen_annotation_code_or_rules_changed:" + relative)
    manifests, manifest_hashes = {}, {}
    for split in SPLITS:
        path = run.parent / (split + ".jsonl")
        if not path.exists():
            path = run.parent / "manifests" / (split + ".jsonl")
        if sha(path) != config["manifest_sha256"][split]:
            raise ValueError("frozen_manifest_changed:" + split)
        manifests[split] = {r["case_id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}
        manifest_hashes[split] = sha(path)
    groups = [set(r) for r in manifests.values()]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("trajectory_split_leakage")
    labels = ontology(json.loads((run / "taxonomy.json").read_text()))
    output.mkdir(parents=True)
    stats = Counter(); cases = Counter(); outcomes = Counter(); selected = []; skipped = []
    with sqlite3.connect((run / "annotations.sqlite3").as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row; con.execute("BEGIN")
        snapshot_calls = con.execute("SELECT calls FROM run_control WHERE id=1").fetchone()[0]
        rows = list(con.execute("SELECT * FROM jobs WHERE status='ready' ORDER BY sequence"))
    files = {(mode, split): (output / (mode + "_" + split + ".jsonl")).open("x", encoding="utf-8")
             for mode in ("single", "history") for split in SPLITS}
    try:
        for row in rows:
            case, split = row["case_id"], row["split"]
            limit = max_train_cases if split == "train" else max_other_cases
            if limit and cases[split] >= limit:
                continue
            try:
                source = json.loads(row["input_json"]); value = json.loads(row["result_json"])
                manifest = manifests[split][case]
                if manifest["task"] != row["task"] or manifest["outcome"] != row["outcome"]:
                    raise ValueError("manifest_binding_mismatch")
                if value["request_metadata"]["full_reannotation_round"] != config["round_id"]:
                    raise ValueError("annotation_from_other_round")
                if value["input_summary"]["source_fingerprint"] != row["source_fingerprint"]:
                    raise ValueError("source_fingerprint_mismatch")
                if value["scoring_summary"]["version"] != A_VERSION:
                    raise ValueError("A_rules_version_mismatch")
                if value["request_metadata"].get("plan_sha256") != value["scoring_summary"]["action"]["plan_sha256"]:
                    raise ValueError("plan_hash_mismatch")
                count = value["input_summary"]["frame_count"]; terminal = count - 1
                points = {p["frame"]: p for p in value["action_progress_points"]}
                valid = {p["frame"]: p for p in value["action_scoring"].get("action_points", [])}
                instruction = source.get("metadata", {}).get("instruction", "")
                if not instruction:
                    raise ValueError("task_instruction_missing")
                video = str(Path(source["paths"]["video_path"]).resolve())
                if not Path(video).is_file():
                    raise ValueError("source_video_missing")
                video_stat = Path(video).stat()
                media_key = digest({"path": video, "bytes": video_stat.st_size, "mtime_ns": video_stat.st_mtime_ns,
                                    "source_fingerprint": row["source_fingerprint"]})
                planned = targets(count); written = 0; buffered = []; unknown_count = 0
                for frame in planned:
                    p = points.get(frame, {}); trusted = valid.get(frame, {})
                    a = p.get("value")
                    a_known = valid_score(a) and trusted.get("progress") == a and trusted.get("unknown_step_ids") == []
                    cited = trusted.get("evidence_frames", [])
                    a_known = a_known and frame in cited and all(type(f) is int and f <= frame for f in cited)
                    t_known = frame == terminal and row["outcome"] in ("success", "failure")
                    unknown_count += int(not a_known)
                    if not a_known and not t_known:
                        continue
                    auxiliary = auxiliary_targets(value, frame, terminal, row["outcome"], labels, exhaustive)
                    for mode in ("single", "history"):
                        sample = {"schema": SCHEMA, "id": case + "@" + str(frame), "case_id": case, "task": row["task"],
                            "split": split, "mode": mode, "frame": frame, "frame_count": count,
                            "frames": causal_frames(range(frame + 1), frame, mode, history_max_frames),
                            "history_max_frames": history_max_frames, "instruction": instruction,
                            "video": video, "media_key": media_key, "video_bytes": video_stat.st_size,
                            "video_mtime_ns": video_stat.st_mtime_ns,
                            "targets": {"A": {"value": float(a) if a_known else 0., "mask": bool(a_known)},
                                        "T": {"value": float(row["outcome"] == "success") if t_known else 0., "mask": t_known},
                                        **auxiliary},
                            "provenance": {"round_id": config["round_id"], "source_fingerprint": row["source_fingerprint"],
                                           "plan_sha256": value["scoring_summary"]["action"]["plan_sha256"],
                                           "labels_are_model_candidates": True}}
                        validate_sample(sample)
                        buffered.append((mode, canonical(sample)))
                    written += 1
                if written:
                    stats["A_unknown_points"] += unknown_count
                    for mode, encoded in buffered:
                        files[(mode, split)].write(encoded + "\n")
                        stats[mode + "_" + split + "_samples"] += 1
                    cases[split] += 1; outcomes[split + "/" + row["outcome"]] += 1
                    selected.append({"case_id": case, "split": split, "source_fingerprint": row["source_fingerprint"]})
            except (ValueError, KeyError, TypeError) as error:
                skipped.append({"case_id": case, "reason": str(error)})
    finally:
        for f in files.values():
            f.close()
    (output / "ontology.json").write_text(canonical(labels) + "\n", encoding="utf-8")
    report = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(), "run": str(run),
              "round_id": config["round_id"], "snapshot_calls": snapshot_calls,
              "history_max_frames": history_max_frames,
              "frozen_annotation_files_sha256": config["frozen_sha256"],
              "original_split_manifest_sha256": manifest_hashes, "ontology_sha256": labels["sha256"],
              "counts": dict(stats), "cases": dict(cases), "outcomes": dict(outcomes), "selected": selected,
              "skipped": skipped, "missing_labels_are_unknown": not exhaustive,
              "test_is_never_training_data": True, "input_fields": ["instruction", "frames", "RGB pixels"],
              "files_sha256": {p.name: sha(p) for p in output.glob("*.jsonl")}}
    (output / "snapshot.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: v for k, v in report.items() if k not in ("selected", "skipped") } | {"skipped_cases": len(skipped)}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--output", required=True)
    p.add_argument("--max-train-cases", type=int, default=0); p.add_argument("--max-other-cases", type=int, default=0)
    p.add_argument("--assume-exhaustive-failure-labels", action="store_true")
    p.add_argument("--history-max-frames", type=int, default=16)
    args = p.parse_args()
    print(json.dumps(export(args.run, args.output, args.max_train_cases, args.max_other_cases,
                            args.assume_exhaustive_failure_labels, args.history_max_frames), ensure_ascii=False))

if __name__ == "__main__":
    main()
