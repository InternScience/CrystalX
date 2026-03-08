# CrystalX Split Repositories

This workspace now contains two cleaned repository-style directories extracted from the original mixed codebase:

- [training_code](c:/Users/kaipe/Downloads/xrd_code/xrd_code/training_code): training, data preparation, and training-side post-processing
- [inference_repo](c:/Users/kaipe/Downloads/xrd_code/xrd_code/inference_repo): inference pipelines, demo scripts, result post-processing, and inference analysis

## Shared Conventions

- Both directories are structured as standalone Python packages with `pyproject.toml`
- Both use package-style imports instead of flat same-directory imports
- Both include a focused `README.md`, `requirements.txt`, and `.gitignore`
- Both preserve a `README.legacy.md` file for older notes
- Python entrypoints are expected to run from the repository root using `python -m ...`

## Package Roots

- Training package: `crystalx_train`
- Inference package: `crystalx_infer`

## External Tools

Both repositories still depend on:

- `SHELXT`
- `SHELXL`
- `PLATON`

## Suggested Next Step

If you want to fully separate them from this workspace, the next clean step is:

1. Initialize a Git repository inside `training_code`
2. Initialize a Git repository inside `inference_repo`
3. Install dependencies per repository with `pip install -r requirements.txt`
4. Verify one representative module entrypoint from each repository
