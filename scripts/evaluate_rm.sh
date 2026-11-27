#!/usr/bin/env bash
set -euo pipefail
: "${FAILATLAS_RM_CHECKPOINT:?Set checkpoint directory}"
: "${FAILATLAS_SNAPSHOT:?Set the identical frozen evaluation snapshot}"
: "${FAILATLAS_EVAL_DIR:?Set a new evaluation output directory}"
: "${FAILATLAS_FRAME_CACHE:?Set frame cache directory}"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -- "${FAILATLAS_EVAL_DIR}"
python -m qwen_rm.predict --checkpoint "${FAILATLAS_RM_CHECKPOINT}" --snapshot "${FAILATLAS_SNAPSHOT}" \
  --split validation --frame-cache "${FAILATLAS_FRAME_CACHE}" --output "${FAILATLAS_EVAL_DIR}/validation.jsonl"
python -m qwen_rm.predict --checkpoint "${FAILATLAS_RM_CHECKPOINT}" --snapshot "${FAILATLAS_SNAPSHOT}" \
  --split test --frame-cache "${FAILATLAS_FRAME_CACHE}" --calibration "${FAILATLAS_EVAL_DIR}/validation.jsonl.metrics.json" \
  --output "${FAILATLAS_EVAL_DIR}/test.jsonl"
python -m failatlas evaluate --split test --predictions "${FAILATLAS_EVAL_DIR}/test.jsonl" \
  --samples "${FAILATLAS_SNAPSHOT}/history_test.jsonl" --ontology "${FAILATLAS_SNAPSHOT}/ontology.json" \
  --calibration "${FAILATLAS_EVAL_DIR}/validation.jsonl.metrics.json" --output "${FAILATLAS_EVAL_DIR}/all_metrics.json"
