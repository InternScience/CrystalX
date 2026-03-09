# CrystalX Workspace

This workspace is organized into three parts:

- `data_pipeline/`: dataset splitting, test-subset collection, SHELX input preparation, and simple batch SHELX execution
- `training_code/`: cleaned training package with the heavy-atom and hydrogen trainers
- `inference_repo/`: cleaned inference package with heavy, hydro, joint, and demo inference entrypoints

## Repository Layout

```text
xrd_code/
  data_pipeline/
  training_code/
  inference_repo/
  environment.yml
  readme.md
```

## Environment

The shared conda environment file is `environment.yml`.

Create the environment with:

```bash
conda env create -f environment.yml
conda activate test_xrd
```

The environment is centered on:

- Python 3.12
- PyTorch 2.4.0 with CUDA 12.1 wheels
- PyTorch Geometric and matching scatter/cluster/sparse extensions
- `cctbx-base`
- `ase`
- `rdkit`
- `scikit-learn`
- `sympy`
- `tqdm`

External binaries still required outside Python:

- `SHELXT`
- `SHELXL`
- `PLATON`

## Data Pipeline

`data_pipeline/` is a standalone utilities directory, grouped into:

- `split/`: build `train_ids`, `test_ids`, and `missing_ids`
- `collect/`: copy selected sample folders and gather CIF files
- `prepare/`: convert CIF to INS, generate or repair HKL files, and assemble SHELX-ready inputs
- `run/`: simple batch SHELXT and SHELXL runners
- `scripts/`: shell wrappers showing the intended step order

Typical flow from the workspace root:

```bash
sh data_pipeline/scripts/01_save_split_ids.sh
sh data_pipeline/scripts/02_collect_test_subset.sh
sh data_pipeline/scripts/03_prepare_shelx_inputs.sh
sh data_pipeline/scripts/04_run_shelx.sh
```

To also run SHELXL in the last step:

```bash
RUN_SHELXL=1 sh data_pipeline/scripts/04_run_shelx.sh
```

## Training

`training_code/` is intentionally trimmed to only the code path needed by its two shell entrypoints:

- `sh training_code/scripts/train/train_heavy.sh`
- `sh training_code/scripts/train/train_hydro.sh`

Equivalent Python entrypoints:

```bash
cd training_code
python -m crystalx_train.trainers.trainer_heavy
python -m crystalx_train.trainers.trainer_hydro
```

The package keeps only:

- `crystalx_train.models`
- `crystalx_train.trainers`

## Inference

`inference_repo/` contains the cleaned inference and demo path. Main shell entrypoints:

- `sh inference_repo/scripts/infer_heavy.sh ...`
- `sh inference_repo/scripts/infer_hydro.sh ...`
- `sh inference_repo/scripts/infer_joint.sh ...`
- `sh inference_repo/scripts/run_demo.sh <file_prefix>`
- `sh inference_repo/scripts/run_demo_batch.sh [ROOT_DIR]`

Equivalent Python entrypoints from inside `inference_repo`:

```bash
python -m crystalx_infer.pipelines.infer_heavy_temporal
python -m crystalx_infer.pipelines.infer_hydro_temporal
python -m crystalx_infer.pipelines.infer_joint_heavy_hydro_temporal
```

The inference package keeps only the modules required by those scripts:

- `crystalx_infer.common`
- `crystalx_infer.models`
- `crystalx_infer.pipelines`
- `crystalx_infer.postprocess`

## Suggested Order

1. Create the conda environment from `environment.yml`.
2. Use `data_pipeline/` to prepare split files and SHELX-ready inputs if needed.
3. Run `training_code/` to train or reproduce heavy/hydro models.
4. Run `inference_repo/` for evaluation, demo inference, or joint prediction.

## Notes

- Each subdirectory has its own focused `README.md`.
- The workspace also contains older legacy scripts at the repository root; they are not the cleaned primary path.
- The current cleaned workflow is centered on `data_pipeline`, `training_code`, and `inference_repo`.
