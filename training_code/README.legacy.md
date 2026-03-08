# CrystalX Training and Data Processing

This file preserves the earlier training-oriented notes in a lightweight form after the repository split.

## Original Scope

- Environment setup with Python 3.12, PyTorch, PyG, `cctbx-base`, `ase`, `rdkit`, and `tqdm`
- Data download and conversion
- Heavy-atom and hydrogen training/evaluation
- Post-training crystallographic processing

## Historical File Mapping

- Data download: `realdata_download.py`
- File organization: `file_organize.py`
- PDB generation: `gen_pdb.py`
- HKL extraction and conversion: `PLATON_cif2hkl.py`, `process_hkl.py`
- INS preparation: `process_cif.py`
- Automatic phasing: `SHELXT_process_final.py`
- Target preparation: `trainer_prepare_all_eval_add.py`
- Heavy training: `trainer_coarse2fine_crystal_add_10fold.py`
- Heavy evaluation: `trainer_coarse2fine_crystal_add_eval.py`
- Hydro training: `trainer_hydro_denoiser_add_10fold.py`
- Hydro evaluation: `trainer_hydro_denoiser_add_eval.py`
- Shared utilities: `crystalx_train/common/utils.py`

Use `README.md` for the current standalone-repository layout.
