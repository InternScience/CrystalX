# CrystalX Inference Repository

Standalone inference repository layout for CrystalX.

## Repository Layout

```text
inference_repo/
  crystalx_infer/
    common/
    models/
    pipelines/
    postprocess/
  scripts/
  weights/
  .gitignore
  pyproject.toml
  requirements.txt
  README.md
  README.legacy.md
```

## Package Layout

- `crystalx_infer.common`: shared utilities used by inference and refinement
- `crystalx_infer.models`: TorchMD-based model definitions
- `crystalx_infer.pipelines`: main inference entrypoints and demo pipelines
- `crystalx_infer.postprocess`: post-inference structure processing helpers
- `scripts`: shell entrypoints for single-case and batch execution
- `weights`: local model checkpoints used by the demo pipeline

## Entrypoints

- Heavy-only inference:
  - `python -m crystalx_infer.pipelines.infer_heavy_temporal`
  - `sh scripts/infer_heavy.sh ...`
- Hydro-only inference:
  - `python -m crystalx_infer.pipelines.infer_hydro_temporal`
  - `sh scripts/infer_hydro.sh ...`
- Joint inference:
  - `python -m crystalx_infer.pipelines.infer_joint_heavy_hydro_temporal`
  - `sh scripts/infer_joint.sh ...`
- Demo pipeline:
  - `sh scripts/run_demo.sh <file_prefix>`
- Batch demo pipeline:
  - `sh scripts/run_demo_batch.sh [ROOT_DIR]`

Run Python commands from the `inference_repo` directory so Python can resolve the `crystalx_infer` package.

## Environment

```bash
conda create -n crystalx_infer python=3.12
conda activate crystalx_infer
conda install -c conda-forge cctbx-base
pip install -r requirements.txt
```

Some PyG-related dependencies may need wheel installation matched to your local PyTorch and CUDA version.

## External Tools

- `SHELXT`
- `SHELXL`
- `PLATON`

## Notes

- Original files under the repository root and the old `inference_code/` directory were not modified during the split.
- Shell scripts now default to the local `weights/` directory instead of old absolute paths.
- Internal imports were rewritten to package-style imports.
- Legacy demo notes are preserved in `README.legacy.md`.
