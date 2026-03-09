# CrystalX Demo Notes

This file preserves the earlier demo-oriented notes after the repository split.

## Original Scope

- Linux x86 environment
- Python 3.12 with PyTorch, PyG, `cctbx-base`, `ase`, and `rdkit`
- External tools: `SHELXT`, `SHELXL`, `PLATON`
- Demo execution through a shell script using local `.ins` and `.hkl` inputs

## Historical Demo Entry

- Single-case demo script: `scripts/run_demo.sh`
- Batch runner: `scripts/run_demo_batch.sh`
- Heavy stage: `crystalx_infer.pipelines.predict_heavy`
- Hydro stage: `crystalx_infer.pipelines.predict_hydro`

Use `README.md` for the current standalone-repository layout.
