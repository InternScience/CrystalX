# CrystalX Training Repository

Standalone training repository layout for CrystalX.

## Repository Layout

```text
training_code/
  crystalx_train/
    common/
    data_prep/
    models/
    postprocess/
    trainers/
  .gitignore
  pyproject.toml
  requirements.txt
  README.md
  README.legacy.md
```

## Package Layout

- `crystalx_train.common`: shared utilities
- `crystalx_train.data_prep`: dataset download, conversion, and label preparation
- `crystalx_train.models`: model definitions and TorchMD components
- `crystalx_train.trainers`: training and evaluation entrypoints
- `crystalx_train.postprocess`: post-training crystallographic processing

## Entrypoints

- Heavy training:
  - `python -m crystalx_train.trainers.trainer_heavy`
  - `sh scripts/train/train_heavy.sh`
- Hydrogen training:
  - `python -m crystalx_train.trainers.trainer_hydro`
  - `sh scripts/train/train_hydro.sh`
- Data preparation:
  - `python -m crystalx_train.data_prep.realdata_download`
  - `python -m crystalx_train.data_prep.process_cif`
  - `python -m crystalx_train.data_prep.process_hkl`
- Post-processing:
  - `python -m crystalx_train.postprocess.SHELXL_process`
  - `python -m crystalx_train.postprocess.weight_refine`

Run commands from the `training_code` directory so Python can resolve the `crystalx_train` package.

## Environment

```bash
conda create -n crystalx_train python=3.12
conda activate crystalx_train
conda install -c conda-forge cctbx-base
pip install -r requirements.txt
```

Some PyG-related dependencies may need wheel installation matched to your local PyTorch and CUDA version.

## External Tools

- `SHELXT`
- `SHELXL`
- `PLATON`

## Notes

- Original files in the repository root were not modified during the split.
- Internal imports were rewritten to package-style imports.
- Relative data and output paths inside the scripts were intentionally left unchanged in this pass.
- Legacy training notes are preserved in `README.legacy.md`.
