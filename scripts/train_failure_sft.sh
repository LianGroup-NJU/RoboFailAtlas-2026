#!/usr/bin/env bash
set -euo pipefail
: "${FAILATLAS_RLINF_ROOT:?Set qualified RLinf/OpenPI}"
: "${FAILATLAS_EXPERT_DATA:?Set LeRobot expert dataset}"
: "${FAILATLAS_SFT_CHECKPOINT:?Set original SFT}"
: "${FAILATLAS_NORM_STATS:?Set normalization statistics}"
: "${FAILATLAS_SFT_WORK:?Set work containing plan.json and datasets/task/anchors.npz}"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_dir}/src:${FAILATLAS_RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec torchrun --standalone --nproc_per_node "${NPROC:-2}" "${repo_dir}/integrations/failure_sft/train.py" --group "${1:?guided or random}" --task "${2:?task}" --run "${3:?new run directory}"
