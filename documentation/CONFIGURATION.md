# Configuration layout

Public entry points use a small number of model and task presets. Detailed
shared defaults are retained for reproducibility, rather than copied into each
preset. No new experiment-management framework is required.

The original RM JSON presets move as follows:

| Original | Current |
|---|---|
| configs/rm/joint_4gpu.json | configs/rm_joint.json |
| configs/rm/task_only_4gpu.json | configs/rm_task_only.json |
| configs/rm/t_only_4gpu.json | configs/rm_t_only.json |
| joint_lambda005_4gpu.json | configs/ablations/struct_005.json |
| joint_lambda020_4gpu.json | configs/ablations/struct_020.json |
| joint_rec025_4gpu.json | configs/ablations/recovery_025.json |
| joint_rec100_4gpu.json | configs/ablations/recovery_100.json |

PPO configs move from configs/ppo/TASK.yaml to configs/tasks/TASK.yaml;
configs/ppo.yaml contains their shared parameters. scripts/train_ppo.sh still
takes the task name. The script writes a complete YAML before Hydra is invoked,
so `_base_` is never passed to the native RLinf runtime. Hydra interpolation and
external defaults remain unchanged. Additional Hydra overrides follow the task.

The reward profile moves to configs/reward.json without changing its bytes.
The SFT plan moves to configs/sft.json without changing its bytes.

To inspect a preset without environment expansion in Python:

```python
from failatlas.config import resolve
config = resolve("configs/rm_joint.json")
```

For a runnable expanded config use `failatlas config INPUT --output NEW_FILE`.
Output files are created exclusively, preventing an existing run's configuration
from being overwritten. Later bases override earlier bases; the child overrides
all bases. Lists are replaced, not concatenated. A cycle or missing base fails.

Training/data/checkpoint/budget values are preserved from the internal source
package. In particular, archived PPO presets contain 20/100-round budgets;
these differ from the author's later paper statement of 50 rounds. This cleanup
does not rewrite either record or assert those old presets reproduce new results.
The common two optimization epochs per PPO update remain unchanged. Aligning
runtime recipes with a newly confirmed experiment requires separate verification.
