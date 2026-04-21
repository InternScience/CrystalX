"""Prepare SHELXL bond-calculation input files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from crystalx_infer.common.paths import BondInputPaths
from crystalx_infer.common.shelx import copy_file


@dataclass
class BondInputConfig:
    fname: str = "sample2_AIhydro"
    work_dir: str = "work_dir"


def run_prepare(config: BondInputConfig) -> None:
    paths = BondInputPaths.from_values(config.work_dir, config.fname)
    copy_file(str(paths.source_hkl_path), str(paths.output_hkl_path))
    copy_file(str(paths.source_res_path), str(paths.output_ins_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare SHELXL bond-calculation inputs")
    parser.add_argument("--fname", type=str, default=BondInputConfig.fname)
    parser.add_argument("--work_dir", type=str, default=BondInputConfig.work_dir)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_prepare(BondInputConfig(**vars(args)))


if __name__ == "__main__":
    main()
