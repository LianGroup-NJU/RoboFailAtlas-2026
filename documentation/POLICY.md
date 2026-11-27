# Policy reproduction

Model roles are separate:
- Offline Joint RM: historical checkpoint23244, for offline prediction tables.
- Structured PPO Joint RM: historical checkpoint11622, for original policy rewards.
- Full T-only RM: separate supplementary control.
- Initial SFT policy: original step40000 and original normalization statistics.

## Prepare the qualified runtime

~~~bash
python scripts/prepare_rlinf.py --source "$FAILATLAS_RLINF_ROOT" \
 --output work/rlinf_structured --backend runtime
export FAILATLAS_RLINF_RUNTIME="$PWD/work/rlinf_structured"
~~~

Source files are hash-checked; the source checkout is never modified.
Use a second output with runtime_legacy for adjust_bottle's original KL=0 backend.
The custom RLinf/OpenPI snapshot and its development-build dependencies still need
portable distribution. No arbitrary upstream commit is claimed equivalent.

## RM and PPO

In the RM environment, with a dedicated GPU visible, run scripts/serve_rm.sh.
In the policy environment export JOINT_RM_URL, JOINT_RL_RUN and paths from
.env.example, then run scripts/train_ppo.sh beat_block_hammer.

The six-GPU resource profile uses four colocated actor/rollout GPUs plus RM services.
Choose physical placement in YAML for the assigned devices.
Source PPO used four actor and eight rollout ranks; this profile uses four
rollout ranks while preserving effective batch64 and24 total environments.
The reward hash is the normalized profile_hash, not a JSON file-byte hash.

The portable template budgets are hammer20, move20, cup20, press100, adjust100.
Press selects20/50/100 using training success only. Tests never select checkpoints.
The original selection functions are in integrations/ppo/training_selection.py.

## Failure-guided SFT

The segmentation/kinematics/clip_training modules retain native expert windows
and masked flow loss. Real target actions outside a selected clip are preserved;
only their losses are masked.

FAILATLAS_SFT_WORK contains plan.json and datasets/task/anchors.npz
with guided, random, valid_steps and clip_ids. Clip membership must match the
original LeRobot expert dataset. Start from configs/sft.json and
use scripts/train_failure_sft.sh.

The three later clip-protocol tasks are adjust/move/cup. Press/hammer reuse
historical branches; the experiment map records this instead of silently
calling every historical run identical. Expert data and clip memberships
belong to the data release.

Native evaluation depends on qualified simulator/task assets and frozen cohort
manifests. Keep100 scene seeds and instructions per task and raw scene outcomes.
Pass integer successes/episodes to aggregate-policy.


## Reported checkpoints versus rerun templates

The main-paper selected Structured checkpoints use press50, hammer20, move20,
cup20, and adjust100 collection rounds. Task reward uses adjust20; environment
reward uses press98 and adjust100. Supplementary Full T-only comparisons use 20 rounds on every task. These are collection rounds, not actor
optimizer-step counts. Some archived evaluation instruction strings also differ.
See documentation/reproduction/experiments.json for the explicit budget map.

Modern runs include ten critic-only warmup rounds and a rollout-KL guard. The
legacy adjust configuration differs. Matching a nominal collection budget does
not imply matching actual actor updates or historical evaluation prompts.
The portable six-GPU layout is a rerun example, not a claim of bitwise replay.

FAILATLAS_ROBOTWIN_ASSETS points to the qualified task assets used by the native
simulator. It is distinct from the Franka RoboTwin source path.
