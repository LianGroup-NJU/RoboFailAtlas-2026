from __future__ import annotations

import math


def auc(pairs):
    positives = sum(y for _, y in pairs); negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    rows = sorted(pairs); rank_sum = 0.; i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and rows[j][0] == rows[i][0]:
            j += 1
        rank_sum += sum(y for _, y in rows[i:j]) * (i + 1 + j) / 2; i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def classification(pairs, threshold):
    if threshold is None or not pairs:
        return {"balanced_accuracy": None, "accuracy": None, "success_f1": None}
    tp = sum(s >= threshold and y == 1 for s, y in pairs); tn = sum(s < threshold and y == 0 for s, y in pairs)
    fp = sum(s >= threshold and y == 0 for s, y in pairs); fn = sum(s < threshold and y == 1 for s, y in pairs)
    return {"balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None,
            "accuracy": (tp + tn) / len(pairs), "success_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def choose_threshold(pairs, split):
    if split != "validation":
        raise ValueError("threshold_selection_requires_validation_split")
    positives = sum(y for _, y in pairs); negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    rows = sorted(pairs); tp, fp = positives, negatives; best = (-1., -math.inf); i = 0
    while i < len(rows):
        threshold = rows[i][0]
        candidate = ((tp / positives + (negatives - fp) / negatives) / 2, threshold)
        best = max(best, candidate); j = i
        while j < len(rows) and rows[j][0] == threshold:
            tp -= rows[j][1]; fp -= 1 - rows[j][1]; j += 1
        i = j
    best = max(best, (.5, math.nextafter(rows[-1][0], math.inf)))
    return best[1]


def summarize(records, split, calibration=None):
    errors = [r["A_prediction"] - r["A_target"] for r in records if r["A_mask"]]
    pairs = [(r["T_prediction"], int(r["T_target"])) for r in records if r["T_mask"]]
    threshold = choose_threshold(pairs, split) if split == "validation" else (calibration or {}).get("T_threshold")
    if split == "test" and calibration is None:
        raise ValueError("test_requires_frozen_validation_calibration")
    return {"split": split, "samples": len(records), "A_reference": "Gemini candidate physical progress",
            "A_valid_points": len(errors), "A_mae": sum(abs(e) for e in errors) / len(errors) if errors else None,
            "A_rmse": math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None,
            "T_reference": "environment terminal outcome", "T_terminal_cases": len(pairs), "T_auc": auc(pairs),
            "T_threshold": threshold, "threshold_source": "validation", **classification(pairs, threshold)}
