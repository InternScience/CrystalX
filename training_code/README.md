# CrystalX Training Repository

Standalone training repository layout for CrystalX.

## Repository Layout

```text
training_code/
  crystalx_train/
    models/
    trainers/
  scripts/
  .gitignore
  pyproject.toml
  requirements.txt
  README.md
  README.legacy.md
```

## Package Layout

- `crystalx_train.models`: model definitions required by the training scripts
- `crystalx_train.trainers`: heavy-atom and hydrogen training entrypoints
- `scripts/train`: shell wrappers for the two trainer modules

## Entrypoints

- Heavy training:
  - `python -m crystalx_train.trainers.trainer_heavy`
  - `sh scripts/train/train_heavy.sh`
- Hydrogen training:
  - `python -m crystalx_train.trainers.trainer_hydro`
  - `sh scripts/train/train_hydro.sh`

Run commands from the `training_code` directory so Python can resolve the `crystalx_train` package.

## Environment

```bash
conda create -n crystalx_train python=3.12
conda activate crystalx_train
conda install -c conda-forge cctbx-base
pip install -r requirements.txt
```

Some PyG-related dependencies may need wheel installation matched to your local PyTorch and CUDA version.

## Notes

- This repository is intentionally trimmed to only the code path required by `scripts/train/train_heavy.sh` and `scripts/train/train_hydro.sh`.
- Relative data and output paths inside the trainers were intentionally left unchanged.
- Legacy training notes are preserved in `README.legacy.md`.
