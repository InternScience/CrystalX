"""Write final CrystalX structure bundles and metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile
from dataclasses import dataclass

from crystalx_infer.common.chem import build_formula
from crystalx_infer.common.paths import FinalOutputPaths
from crystalx_infer.common.shelx import (
    copy_file,
    get_R2,
    load_shelxt_final,
    read_checkcif,
    update_shelxt_final,
)


@dataclass
class FinalOutputConfig:
    fname: str = "sample2_AIhydroWeight"
    work_dir: str = "work_dir"


def write_xyz(xyz_save_path, atom_symbols, real_cart, formula):
    with open(xyz_save_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"{len(atom_symbols)}\n")
        file_obj.write(f"{formula}\n")
        for atom, coord in zip(atom_symbols, real_cart):
            file_obj.write(f"{atom} {coord[0]} {coord[1]} {coord[2]}\n")


def write_gjf(gjf_save_path, atom_symbols, real_cart, formula):
    with open(gjf_save_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("%chk=11.chk\n")
        file_obj.write("#opt freq pm6\n\n")
        file_obj.write(f"Title {formula}\n\n")
        file_obj.write("0 1\n")
        for atom, coord in zip(atom_symbols, real_cart):
            file_obj.write(f"{atom} {coord[0]} {coord[1]} {coord[2]}\n")


def run_write(config: FinalOutputConfig) -> None:
    from cctbx.xray.structure import structure
    from iotbx.shelx import crystal_symmetry_from_ins

    paths = FinalOutputPaths.from_values(config.work_dir, config.fname)
    source_res = paths.source_res_path
    is_structure_valid = False
    is_quality_valid = False

    if (not source_res.exists() or source_res.stat().st_size == 0) and paths.fallback_res_path.exists():
        source_res = paths.fallback_res_path
        zip_list = [
            str(paths.final_ins_path),
            str(paths.final_hkl_path),
            str(paths.final_xyz_path),
            str(paths.final_gjf_path),
        ]
    else:
        if paths.source_chk_path.exists():
            paths.source_chk_path.replace(paths.final_chk_path)
            is_structure_valid, is_quality_valid = read_checkcif(str(paths.final_chk_path))
        zip_list = [
            str(paths.final_ins_path),
            str(paths.final_hkl_path),
            str(paths.final_cif_path),
            str(paths.final_xyz_path),
            str(paths.final_gjf_path),
            str(paths.final_chk_path),
        ]

    real_frac, atom_symbols = load_shelxt_final(str(source_res), begin_flag="FVAR")
    atom_symbols = [item.capitalize() for item in atom_symbols]
    formula = build_formula(atom_symbols)

    update_shelxt_final(
        str(source_res),
        str(paths.final_ins_path),
        dict(Counter(atom_symbols)),
        no_given_sfac=True,
    )
    copy_file(str(paths.source_hkl_path), str(paths.final_hkl_path))
    if paths.source_cif_path.exists():
        copy_file(str(paths.source_cif_path), str(paths.final_cif_path))

    crystal_symmetry = crystal_symmetry_from_ins.extract_from(str(source_res))
    crystal_structure = structure(crystal_symmetry=crystal_symmetry)
    unit_cell = crystal_structure.crystal_symmetry().unit_cell()
    real_cart = [list(unit_cell.orthogonalize(point)) for point in real_frac]

    write_xyz(str(paths.final_xyz_path), atom_symbols, real_cart, formula)
    write_gjf(str(paths.final_gjf_path), atom_symbols, real_cart, formula)

    with zipfile.ZipFile(paths.final_zip_path, "w") as zip_obj:
        for path in zip_list:
            if path and Path(path).exists():
                zip_obj.write(path)

    wr2, r1, goof, q1 = get_R2(str(source_res))
    fitting_metrics = {
        "wr2": wr2,
        "r1": r1,
        "goof": goof,
        "q1": q1,
        "formula": formula,
        "formular": formula,
        "is_structure_valid": is_structure_valid,
        "is_quality_valid": is_quality_valid,
        "structure_mes": (
            "Stucture valid!"
            if is_structure_valid
            else "Slight structural flaws, possibly due to disorder or the ambiguity of hydrogen atom positions."
        ),
        "quality_mes": "Reflections valid!" if is_quality_valid else "May need reflection correction",
    }
    with open(paths.final_metrics_path, "w", encoding="utf-8") as file_obj:
        json.dump(fitting_metrics, file_obj)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write final XYZ/GJF/metrics bundle")
    parser.add_argument("--fname", type=str, default=FinalOutputConfig.fname)
    parser.add_argument("--work_dir", type=str, default=FinalOutputConfig.work_dir)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_write(FinalOutputConfig(**vars(args)))


if __name__ == "__main__":
    main()
