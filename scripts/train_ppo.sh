#!/usr/bin/env bash
set -euo pipefail
: "${FAILATLAS_RLINF_ROOT:?Set qualified RLinf source}"
: "${FAILATLAS_RLINF_RUNTIME:?Run prepare_rlinf.py first}"
: "${FAILATLAS_ROBOTWIN_ROOT:?Set qualified Franka RoboTwin}"
: "${FAILATLAS_SIM_PYTHON:?Set simulator Python}"
: "${FAILATLAS_SFT_CHECKPOINT:?Set original SFT}"
: "${FAILATLAS_NORM_STATS:?Set normalization statistics}"
: "${FAILATLAS_PPO_RM_CHECKPOINT:?Set original PPO Joint RM}"
: "${JOINT_RM_URL:?Set frozen RM localhost URL}"
: "${JOINT_RL_RUN:?Set a new output directory}"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task="${1:?Pass task name}"
case "${task}" in press_stapler|beat_block_hammer|move_can_pot|place_empty_cup|adjust_bottle) ;; *) exit 2 ;; esac
export PYTHONPATH="${FAILATLAS_RLINF_RUNTIME}:${repo_dir}/integrations/ppo:${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export REPO_PATH="${FAILATLAS_RLINF_ROOT}" EMBODIED_PATH="${FAILATLAS_RLINF_ROOT}/examples/embodiment"
export EXT_MODULE=joint_rl.extension RLINF_CODE_WORKING_DIR=0
export JOINT_RL_FRANKA_REPO="${FAILATLAS_ROBOTWIN_ROOT}"
export JOINT_STRUCTURED_PROFILE="${repo_dir}/configs/reward.json"
export JOINT_STRUCTURED_PROFILE_SHA256="$(python -c 'import os,json;from joint_rl.structured_reward import profile_hash;print(profile_hash(json.load(open(os.environ["JOINT_STRUCTURED_PROFILE"]))))')"
export FAILATLAS_PPO_RM_FINGERPRINT="$(python -c 'import os;from pathlib import Path;from qwen_rm.checkpoints import model_fingerprint;print(model_fingerprint(Path(os.environ["FAILATLAS_PPO_RM_CHECKPOINT"]))))')"
export RAY_ADDRESS=local RAY_TMPDIR="$(mktemp -d /tmp/fa-ray-XXXXXX)"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 WANDB_MODE=disabled JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p -- "${JOINT_RL_RUN}/config"
resolved_dir="$(cd -- "${JOINT_RL_RUN}/config" && pwd)"
python -m failatlas config "${repo_dir}/configs/tasks/${task}.yaml" --output "${resolved_dir}/${task}.yaml"
exec python -B -u "${repo_dir}/integrations/ppo/train_entry.py" --config-path "${resolved_dir}" --config-name "${task}" "${@:2}"
