#!/usr/bin/env bash
set -euo pipefail
: "${FAILATLAS_BASE_MODEL:?Set the pinned local Qwen3-VL model directory}"
: "${FAILATLAS_SNAPSHOT:?Set the training snapshot directory}"
: "${FAILATLAS_RUN:?Set a new output run directory}"
: "${FAILATLAS_FRAME_CACHE:?Set a decoded-frame cache directory}"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
config="${1:-${repo_dir}/configs/rm_joint.json}"
case "${config}" in
  joint) config="${repo_dir}/configs/rm_joint.json" ;;
  task-only) config="${repo_dir}/configs/rm_task_only.json" ;;
  t-only) config="${repo_dir}/configs/rm_t_only.json" ;;
esac
resolved="${FAILATLAS_RUN}.resolved.json"
python -m failatlas config "${config}" --output "${resolved}"
accelerate launch --num_processes "${NPROC:-4}" -m qwen_rm.train \
  --config "${resolved}" --snapshot "${FAILATLAS_SNAPSHOT}" \
  --output "${FAILATLAS_RUN}" --frame-cache "${FAILATLAS_FRAME_CACHE}"
