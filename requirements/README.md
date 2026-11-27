# Environments

RM: Python3.11.17, Torch2.8.0+cu128, Transformers4.57.6.
Policy: Python3.11.15, Torch2.6.0, Transformers4.53.2.
Simulator: qualified RoboTwin/Franka environment separately.

The version lists are observed package versions, not complete lockfiles.
Do not combine both environments. Development builds and the custom RLinf/OpenPI
source require distributable snapshots. The doctor command never initializes CUDA.
Core tests use standard Python; optional SFT/loss tests use NumPy/Torch.
