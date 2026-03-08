# CrystalX - Training and Data Processing

## 1. Environment Preparation

### Device
- A Linux x86 platform
- Single RTX 4090 GPU with CUDA Version: 12.2

### Python Environment Installation
Set up the Python environment by following these steps:

```bash
# Create a new conda environment with Python 3.12
conda create -n test_xrd python=3.12

# Activate the new environment
conda activate test_xrd

# Install cctbx-base using conda
conda install -c conda-forge cctbx-base

# Install PyTorch and related packages
pip install torch
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Install additional dependencies
pip install ase
pip install rdkit
pip install tqdm
pip install torch_geometric
```

### Required Binaries
The following binary programs are required:
- **SHELXT**
- **SHELXL**
- **PLATON**

They can all be downloaded freely from their respective official websites.

## 2. Data Preparation

### Data Download
- Download real data: `realdata_download.py`

### File Organization & Format Conversion
- Organize downloaded files: `file_organize.py`
- Generate PDB formats: `gen_pdb.py`

### Extract Diffraction Data
- Extract embedded diffraction data from CIF files using PLATON: `PLATON_cif2hkl.py`

### HKL File Processing
- Convert individual **HKL** files to the SHELX suite "HKL 4" supported format: `process_hkl.py`

### INS File Preparation
- Write CIF files into initial INS files: `process_cif.py`

### Automatic Phasing
- Run automatic phasing with SHELXT using the prepared INS and HKL files: `SHELXT_process_final.py`

### Target matching
- Match phasing results with human experts’ analysis: `trainer_prepare_all_eval_add.py`

> **Note**: The dataset consists of 51,433 entries, located in `all_cif`, with the annotated coarse electron density available at `all_anno_density`.

---

## 3. Models

The following models are utilized in the training:

- **SchNet**: `schnet.py`
- **ComENet**: `comenet.py`
- **DimeNet**: `dimenet.py`
- **SphereNet**: `spherenet.py`
- **TorchMD-Net**: 
  - `torchmd_net.py`
  - `torchmd_et.py`
  - `torchmd_utils.py`
  - `noise_output_model.py`

---

## 4. Training & Evaluation

### Heavy Elemental Determination

- **Training**: `trainer_coarse2fine_crystal_add_10fold.py`
- **Evaluation**: `trainer_coarse2fine_crystal_add_eval.py`

### Hydrogen Atom Determination

- **Training**: `trainer_hydro_denoiser_add_10fold.py`
- **Evaluation**: `trainer_hydro_denoiser_add_eval.py`

---

## 5. Crystallographic Computing & Error Correction

- **Run automatic SHELXL processing**: `SHELXL_process.py`
- **Run automatic weight refinement**: `weight_refine.py`
- **Crystallographic comparison for error correction**:
  - Main comparison: `wr2_compare_main.py`
  - Hydrogen comparison: `wr2_compare_hydro.py`

---

## 6. Utility Functions

- Various utility functions: `utils.py`


