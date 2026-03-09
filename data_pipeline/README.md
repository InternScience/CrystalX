# Data Pipeline

This directory contains standalone scripts for dataset splitting, test-subset collection, SHELX input preparation, and simple SHELX execution.

## Layout

```text
data_pipeline/
  README.md
  scripts/
    01_save_split_ids.sh
    02_collect_test_subset.sh
    03_prepare_shelx_inputs.sh
    04_run_shelx.sh
    save_split_ids.sh
  split/
    save_split_ids.py
  collect/
    copy_dirs.py
    get_cifs.py
  prepare/
    cif2ins.py
    platon_cif2hkl.py
    check_empty_sxhkl.py
    download_hkl.py
    platon_hkl2hkl.py
    copy_inshkl.py
  run/
    shelxt_simple.py
    shelxl_simple.py
```

## What Each Group Does

- `split/`
  Builds `train_ids`, `test_ids`, and `missing_ids` text files from the year list and pt directory.
- `collect/`
  Copies selected sample directories and gathers matching CIF files for the test subset.
- `prepare/`
  Converts CIF to INS, generates or repairs HKL inputs, and assembles clean SHELX-ready folders.
- `run/`
  Runs simple batch SHELXT or SHELXL jobs over prepared directories.
- `scripts/`
  Shell wrappers that show the intended step-by-step processing flow.

## Typical Flow

1. Save train/test split ids.
2. Copy the selected test subset.
3. Prepare SHELX input files.
4. Run SHELXT, and optionally SHELXL.

Example:

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

## Notes

- These scripts are not packaged as a Python module. They are plain utilities driven by file layout and naming conventions.
- Default paths are relative and meant to be adjusted through script variables or environment variables.
- External tools such as `platon`, `shelxt`, and `shelxl` are expected to be available where the scripts point to them.
