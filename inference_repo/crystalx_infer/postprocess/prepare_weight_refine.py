"""Prepare SHELXL weight-refinement inputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from crystalx_infer.common.paths import WeightRefinePaths
from crystalx_infer.common.shelx import copy_file, update_shelxt_weight


@dataclass
class WeightRefineConfig:
    fname: str = "sample2_AIhydro"
    work_dir: str = "work_dir"


def run_prepare(config: WeightRefineConfig) -> None:
    paths = WeightRefinePaths.from_values(config.work_dir, config.fname)
    res_source = paths.source_res_path
    if (not res_source.exists() or res_source.stat().st_size == 0) and paths.fallback_res_path.exists():
        res_source = paths.fallback_res_path

    copy_file(str(paths.source_hkl_path), str(paths.output_hkl_path))
    update_shelxt_weight(
        str(res_source),
        str(paths.output_ins_path),
        is_weight=True,
        is_acta=True,
        re_afix=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adjust SHELXL weight refinement inputs")
    parser.add_argument("--fname", type=str, default=WeightRefineConfig.fname)
    parser.add_argument("--work_dir", type=str, default=WeightRefineConfig.work_dir)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_prepare(WeightRefineConfig(**vars(args)))


if __name__ == "__main__":
    main()
