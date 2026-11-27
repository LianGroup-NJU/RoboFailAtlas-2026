#!/usr/bin/env bash
set -euo pipefail
: "${FAILATLAS_PPO_RM_CHECKPOINT:?Set frozen PPO RM}"
: "${FAILATLAS_BASE_MODEL:?Set pinned local base model}"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_dir}/integrations/ppo:${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
fingerprint="$(python -c 'import os;from pathlib import Path;from qwen_rm.checkpoints import model_fingerprint;print(model_fingerprint(Path(os.environ["FAILATLAS_PPO_RM_CHECKPOINT"]))))')"
exec python -B -u -m joint_rl.joint_rm_server --checkpoint "${FAILATLAS_PPO_RM_CHECKPOINT}" --checkpoint-sha256 "${fingerprint}" --port "${PORT:-8001}"
