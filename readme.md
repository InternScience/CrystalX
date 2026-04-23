# CrystalX: High-accuracy Crystal Structure Analysis Using Deep Learning

**Accepted by the *Journal of the American Chemical Society* (JACS); invited for a journal cover feature**


## Overview

CrystalX is the first AI system for routine single-crystal structure analysis from real experimental X-ray diffraction (XRD) data.

Designed specifically for everyday single-crystal structure solution, CrystalX uses geometric deep learning to model electron density and capture underlying three-dimensional geometric interactions directly from large-scale experimental XRD datasets. Compared with traditional rule-based approaches for automatic elemental determination, such as those used in **SHELXT** and **Olex2**, CrystalX delivers substantially improved accuracy and robustness.

In prospective, deployment-style evaluations, CrystalX was also compared with **AutoChem** under practical experimental conditions. Because AutoChem requires a real instrument-generated metadata file (`.cif_od`) produced by the **CrysAlisPro** data-reduction workflow, the comparison was performed on real-world cases that satisfied this requirement. CrystalX successfully solved **3/3** test cases, whereas AutoChem solved **1/3**.

CrystalX provides the following capabilities:

* Accurate discrimination between non-hydrogen atoms with similar atomic numbers, including challenging pairs such as **C/N/O** and **P/S/Cl**
* Fast and fully correct solution of large organometallic structures containing up to **370 non-hydrogen atoms**
* Detection of **9 verified expert interpretation errors** among **1,559** held-out structures published in **JCR Q1 journals**, including subtle cases that triggered no **CheckCIF A/B** alerts
* Confidence scores for both heavy-atom and hydrogen predictions
* Natural integration into standard crystallographic workflows


---

## Model Architecture

CrystalX adopts a two-stage geometric deep learning pipeline to predict both non-hydrogen and hydrogen atoms.

Both public checkpoints are built on an Equivariant Transformer backbone, specifically TorchMD-NET.

For hydrogen prediction, CrystalX leverages both intramolecular and intermolecular context by incorporating symmetry-equivalent neighbors within 3.2 Å. This design yields more than a 7% improvement over using intramolecular information alone.

---

## Available Checkpoints

The trained checkpoints are hosted on Hugging Face in the model repository.

Model repo:

- `Kaipengm2/CrystalX`
- `https://huggingface.co/Kaipengm2/CrystalX`

Recommended public checkpoint filenames:

- `crystalx-heavy.pth`
- `crystalx-hydro.pth`

By default, checkpoints are cached locally in:

- `inference_repo/weights/`

To download the weights, run:

```bash
cd inference_repo
python -m crystalx_infer.tools.download_weights --repo-id Kaipengm2/CrystalX
```

---

## Intended Use

In practice, CrystalX can be inserted at different stages of the pipeline for both **heavy-atom** and **hydrogen** prediction seamlessly. The official codebase provides a lightweight integration with the **SHELX** suite, enabling a simple **`.res`-to-`.res`** workflow.

### Current Limitation: Disorder

CrystalX does not currently support the resolution of crystallographic disorder, largely because high-quality annotated training data for these cases are scarce. At the same time, disorder prediction is closely connected to the accurate detection and interpretation of residual electron density, making it a natural future extension of the current framework.

We view disorder modeling as a particularly promising direction for further development. Interpreting disorder is inherently a sequential, multi-step reasoning task: it involves iterative analysis, hypothesis generation, testing, and refinement rather than a single-pass prediction. In this context, agentic AI and reinforcement learning may offer a compelling path forward, as they could enable models to learn from sequential refinement processes and better capture the stepwise reasoning needed for robust disorder resolution.

---

## Minimal End-to-End Workflow

A typical wrapper pipeline is:

`SHELXT -> CrystalX Heavy -> SHELXL refinement -> CrystalX Hydro -> HFIX/AFIX placement -> SHELXL refinement -> weight refinement -> PLATON / CheckCIF`

1. **SHELXT** generates coarse electron-density peaks.
2. **CrystalX Heavy** predicts non-hydrogen atom types from geometric peak interactions.
3. **SHELXL** refines the heavy-atom framework.
4. **CrystalX Hydro** predicts how many hydrogens are attached to each heavy atom.
5. **HFIX/AFIX** placement and subsequent refinement produce the final all-atom structure.

Demo: `https://crystalx.intern-ai.org.cn/`



## Environment Setup

Create the shared environment from [environment.yml](environment.yml):

```bash
conda env create -f environment.yml
conda activate crystalx
```

To run a minimal end-to-end workflow, the following external crystallographic binaries are required:

- `SHELXT`
- `SHELXL`
- `PLATON`


## Quick Use in a SHELX Workflow

CrystalX can be inserted at different stages of the pipeline for both **heavy-atom** and **hydrogen** prediction seamlessly. 

Run inference from `inference_repo/`.

Single-case prediction:

```bash
cd inference_repo
python -m crystalx_infer.pipelines.predict_heavy --help
python -m crystalx_infer.pipelines.predict_hydro --help
```

These two entrypoints expose the individual learned stages used inside the full CrystalX wrapper.

- `predict_heavy` can now run from a single `*.res` file via `--res_path <case>.res`. It reads atom names, fractional coordinates, symmetry, and `SFAC/UNIT` directly from that `.res`, predicts heavy-atom element identities, and writes `<case>_AI.ins`. If you also pass `--hkl_path <case>.hkl`, it additionally copies that file to `<case>_AI.hkl`.
- `predict_heavy` also saves a per-atom probability report `<fname>_AI_topk.json`. By default it stores the top `5` candidate elements per atom; use `--topk K` to change this. 
- `predict_hydro` can now run from a single `*.res` file via `--res_path <case>.res`. It reads the refined heavy-atom template directly from that `.res`, predicts hydrogen counts per heavy atom, writes `<case>hydro.ins`, and if you also pass `--hkl_path <case>.hkl`, it copies that file to `<case>hydro.hkl`.
- `predict_hydro` also outputs per-atom prediction probabilities in `<fname>hydro_topk.json`. These are softmax probabilities saved as top-k entries per atom; use `--topk K` to change how many classes are stored.
- `predict_hydro` will use `<case>Bond.lst` or `<case>.lst` from the same directory when present. If neither exists, it falls back to distance-based connectivity built from the supplied `.res`: it uses RDKit covalent radii to decide whether two atoms are bonded, with the approximate rule `distance <= (r_i + r_j) * 1.20` and `distance > 0.10 A`.
- In the default naming convention used by `run_demo.sh`, the heavy stage is called with `--fname <case>` and produces `<case>_AI.*`; the hydrogen stage is then called with `--fname <case>_AI` and produces `<case>_AIhydro.*`.

End-to-end demo wrapper:

```bash
sh scripts/run_demo.sh <file_prefix>
```
This wrapper corresponds to the minimal wrapper: shelxt, heavy-atom prediction, `SHELXL` refinement, hydrogen prediction, riding-model placement, additional refinement, and final output packaging.


## Dataset

The dataset is constructed from the open-access Crystallography Open Database (COD) using real experimental diffraction data. To ensure reliable supervision, only structures whose final refined solutions can be confidently matched to their corresponding initial phasing results are retained. The resulting dataset contains over 50,000 structures, spanning organic, organometallic, and inorganic crystals, and covers 83 elements and 86 space groups.


Training and dataset-level evaluation use preprocessed `equiv_*.pt` records plus a plain-text split file.

Each `equiv_*.pt` file is a Python dictionary saved with `torch.save`.

Heavy-atom training / heavy evaluation require at least:

```python
{
  "z": ["C", "N", "O", ...],              # input element symbols aligned with pos
  "gt": ["C", "N", "O", ...],             # target heavy-atom labels for the main atoms
  "pos": np.ndarray,                      # shape [N, 3], Cartesian coordinates in angstrom
}
```

Hydrogen training / hydrogen evaluation require at least:

```python
{
  "equiv_gt": ["C", "N", "O", ...],       # main atoms + symmetry-expanded neighbor atoms
  "gt": ["C", "N", "O", ...],             # main heavy atoms only
  "hydro_gt": [0, 1, 2, ...],             # attached H count per main heavy atom
  "pos": np.ndarray,                      # shape [N, 3], Cartesian coordinates in angstrom
}
```

Joint evaluation expects the union of both schemas:

```python
{
  "z": ...,
  "equiv_gt": ...,
  "gt": ...,
  "hydro_gt": ...,
  "pos": ...,
}
```

Important conventions used by the loaders:

- `pos[i]` is aligned with the symbol entry at the same index, such as `z[i]` or `equiv_gt[i]`.
- Element symbols must be valid RDKit-readable symbols, for example `C`, `N`, `O`, `Cl`, `Br`, `Zn`.
- `gt` contains only the main non-hydrogen atoms used as labels.
- `hydro_gt[j]` is the hydrogen count attached to `gt[j]`; hydrogen coordinates themselves are not stored in the training target.
- Duplicate Cartesian coordinates are removed during loading. After deduplication, the first `len(gt)` atoms are assumed to be the main atoms, and any remaining atoms are treated as symmetry-expanded context atoms.


## Training and Evaluation

Run training from `training_code/`.

```bash
cd training_code
python -m crystalx_train.trainers.trainer_heavy --help
python -m crystalx_train.trainers.trainer_hydro --help
```

Shell wrappers are also provided:

```bash
sh scripts/train/train_heavy.sh
sh scripts/train/train_hydro.sh
```

Dataset-level evaluation:

```bash
python -m crystalx_infer.pipelines.infer_heavy_temporal --help
python -m crystalx_infer.pipelines.infer_hydro_temporal --help
python -m crystalx_infer.pipelines.infer_joint_heavy_hydro_temporal --help
```

## Citation

If you find this repository useful, please cite:

```bibtex
@article{doi:10.1021/jacs.5c21832,
  author  = {Zheng, Kaipeng and Huang, Weiran and Ouyang, Wanli and Zhong, Han-Sen and Li, Yuqiang},
  title   = {CrystalX: High-Accuracy Crystal Structure Analysis Using Deep Learning},
  journal = {Journal of the American Chemical Society},
  volume  = {0},
  number  = {0},
  pages   = {null},
  year    = {0},
  doi     = {10.1021/jacs.5c21832}
}
```
