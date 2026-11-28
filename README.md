# RoboFailAtlas

**Structured Reward Learning from Diverse Robot Failures**

[Project website](https://liangroup-nju.github.io/RoboFailAtlas-2026/) ·
[Dataset](https://huggingface.co/datasets/LianGroup-NJU/RoboFailAtlas) ·
[Joint reward model](https://huggingface.co/LianGroup-NJU/RoboFailAtlas-Joint-RM) ·
[Qwen3-VL-4B-Instruct base model](https://huggingface.co/LianGroup-NJU/Qwen3-VL-4B-Instruct)

This repository contains reward-model training and evaluation, policy-learning
adapters, and the project website.

The dataset repository includes the [structured annotations](https://huggingface.co/datasets/LianGroup-NJU/RoboFailAtlas/tree/main/annotations)
and [trajectory upload progress](https://huggingface.co/datasets/LianGroup-NJU/RoboFailAtlas/blob/main/upload-status.json).

## Quick start

Use Python 3.11 or later. Keep reward-model and policy environments separate;
see [environment notes](requirements/README.md).

```bash
python -m pip install -e ".[config]"
python scripts/replay_metrics.py
python -m unittest discover -s tests -v
failatlas --help
```

Fill the paths in `.env.example`, then export the variables needed by your run.
Training requires external datasets, model checkpoints, and the qualified
RLinf/OpenPI/RoboTwin runtime.

```bash
bash scripts/train_rm.sh joint
bash scripts/train_rm.sh task-only
bash scripts/train_rm.sh t-only
bash scripts/evaluate_rm.sh
bash scripts/serve_rm.sh
bash scripts/train_ppo.sh beat_block_hammer
bash scripts/train_failure_sft.sh guided move_can_pot /path/to/new/run
```

## Layout

- `src/`: reward models, data interfaces, and evaluation.
- `configs/`: shared defaults, model variants, and task presets.
- `scripts/`: training, evaluation, serving, and metric-replay entry points.
- `integrations/`: PPO and failure-guided SFT adapters.
- `examples/` and `tests/`: examples, saved-prediction replay, and CPU checks.
- `documentation/`: [data](documentation/DATA.md), [metrics](documentation/METRICS.md),
  [policy runtime](documentation/POLICY.md), and [configuration notes](documentation/CONFIGURATION.md).
- `docs/`: static project website, figures, videos, and trajectory examples.

Presets inherit shared settings through `_base_`. Inspect a resolved config with
`failatlas config configs/rm_joint.json`. The PPO presets retain historical
budgets; their relationship to later paper settings is recorded in the
configuration notes. CPU checks and saved-prediction replay do not establish
end-to-end GPU reproduction.

## Website

The website requires no build step. Preview it from this repository:

```bash
python -m http.server 8000 --directory docs
```

Open <http://localhost:8000>. GitHub Pages serves the website from this
repository's `main` branch and `/docs` directory. Push website changes under
`docs/` to update the live site automatically. Confirmed resource links are
configured in [`docs/release.json`](docs/release.json).

## License and resources

A project license has not yet been selected; see [license status](LICENSE_STATUS.md)
and [third-party notices](third_party/NOTICE.md). The bundled font retains its
[SIL Open Font License](docs/assets/fonts/OFL.txt).

Model weights and external training datasets are not bundled. The confidential
annotation-generation pipeline, full generation prompts, task-scoring
implementation, and private annotation services are excluded.
