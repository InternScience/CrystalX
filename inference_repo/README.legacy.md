# CrystalX Demo Notes

This file preserves the earlier demo-oriented notes after the repository split.

## Original Scope

- Linux x86 environment
- Python 3.12 with PyTorch, PyG, `cctbx-base`, `ase`, and `rdkit`
- External tools: `SHELXT`, `SHELXL`, `PLATON`
- Demo execution through a shell script using local `.ins` and `.hkl` inputs

## Historical Demo Entry

- Single-case demo script: `scripts/temp_demo_new.sh`
- Batch runner: `scripts/batch_run_all.sh`
- Heavy stage: `crystalx_infer.pipelines.temp_demo_new_main`
- Hydro stage: `crystalx_infer.pipelines.temp_demo_new_hydro`

Use `README.md` for the current standalone-repository layout.
