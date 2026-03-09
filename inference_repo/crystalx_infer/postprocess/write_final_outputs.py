import argparse
import json
import os
import zipfile
from collections import Counter

from cctbx.xray.structure import structure
from iotbx.shelx import crystal_symmetry_from_ins

from crystalx_infer.common.utils import (
    copy_file,
    get_R2,
    load_shelxt_final,
    read_checkcif,
    update_shelxt_final,
)


def build_formula(atom_symbols):
    formula = ""
    for symbol, count in Counter(atom_symbols).items():
        formula += f"{symbol}{count}"
    return formula


def write_xyz(xyz_save_path, atom_symbols, real_cart, formula):
    with open(xyz_save_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"{len(atom_symbols)}\n")
        file_obj.write(f"{formula}\n")
        for atom, coord in zip(atom_symbols, real_cart):
            file_obj.write(f"{atom} {coord[0]} {coord[1]} {coord[2]}\n")


def write_gjf(gjf_save_path, atom_symbols, real_cart, formula):
    with open(gjf_save_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("%chk=11.chk\n")
        file_obj.write("#opt freq pm6\n")
        file_obj.write("\n")
        file_obj.write(f"Title {formula}\n")
        file_obj.write("\n")
        file_obj.write("0 1\n")
        for atom, coord in zip(atom_symbols, real_cart):
            file_obj.write(f"{atom} {coord[0]} {coord[1]} {coord[2]}\n")


def main(args):
    qpeak_res_name = f"{args.work_dir}/{args.fname}.res"
    hkl_file_path = f"{args.work_dir}/{args.fname}.hkl"
    cif_file_path = f"{args.work_dir}/{args.fname}.cif"
    chk_file_path = f"{args.work_dir}/{args.fname}.chk"

    final_qpeak_res_name = f"{args.work_dir}/{args.fname}Final.ins"
    new_hkl_file_path = f"{args.work_dir}/{args.fname}Final.hkl"
    new_cif_file_path = f"{args.work_dir}/{args.fname}Final.cif"
    xyz_save_path = f"{args.work_dir}/{args.fname}Final.xyz"
    gjf_save_path = f"{args.work_dir}/{args.fname}Final.gjf"
    metrics_save_path = f"{args.work_dir}/{args.fname}FinalMetrics.json"
    zip_save_path = f"{args.work_dir}/{args.fname}Final.zip"
    new_chk_file_path = f"{args.work_dir}/{args.fname}Final_checkcif.chk"

    if os.path.getsize(qpeak_res_name) == 0:
        qpeak_res_name = f"{args.work_dir}/{args.fname[:-6]}.res"
        is_structure_valid = False
        is_quality_valid = False
        zip_list = [final_qpeak_res_name, new_hkl_file_path, xyz_save_path, gjf_save_path]
    else:
        os.rename(chk_file_path, new_chk_file_path)
        is_structure_valid, is_quality_valid = read_checkcif(new_chk_file_path)
        zip_list = [
            final_qpeak_res_name,
            new_hkl_file_path,
            new_cif_file_path,
            xyz_save_path,
            gjf_save_path,
            new_chk_file_path,
        ]

    real_frac, shelxt_pred = load_shelxt_final(qpeak_res_name, begin_flag="FVAR")
    shelxt_pred = [item.capitalize() for item in shelxt_pred]
    print(shelxt_pred)

    formula = build_formula(shelxt_pred)
    update_shelxt_final(qpeak_res_name, final_qpeak_res_name, dict(Counter(shelxt_pred)), no_given_sfac=True)
    copy_file(hkl_file_path, new_hkl_file_path)
    copy_file(cif_file_path, new_cif_file_path)

    crystal_symmetry = crystal_symmetry_from_ins.extract_from(qpeak_res_name)
    crystal_structure = structure(crystal_symmetry=crystal_symmetry)
    unit_cell = crystal_structure.crystal_symmetry().unit_cell()
    real_cart = [list(unit_cell.orthogonalize(point)) for point in real_frac]

    write_xyz(xyz_save_path, shelxt_pred, real_cart, formula)
    write_gjf(gjf_save_path, shelxt_pred, real_cart, formula)

    with zipfile.ZipFile(zip_save_path, "w") as zip_obj:
        for path in zip_list:
            zip_obj.write(path)

    wr2, r1, goof, q1 = get_R2(qpeak_res_name)
    fitting_metrics = {
        "wr2": wr2,
        "r1": r1,
        "goof": goof,
        "q1": q1,
        "formular": formula,
        "is_structure_valid": is_structure_valid,
        "is_quality_valid": is_quality_valid,
    }
    if is_structure_valid:
        fitting_metrics["structure_mes"] = "Stucture valid!"
    else:
        fitting_metrics["structure_mes"] = (
            "Slight structural flaws, possibly due to disorder or the ambiguity of hydrogen atom positions."
        )
    if is_quality_valid:
        fitting_metrics["quality_mes"] = "Reflections valid!"
    else:
        fitting_metrics["quality_mes"] = "May need reflection correction"

    with open(metrics_save_path, "w", encoding="utf-8") as file_obj:
        json.dump(fitting_metrics, file_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write final XYZ/GJF/metrics bundle")
    parser.add_argument("--fname", type=str, default="sample2_AIhydroWeight", help="file name")
    parser.add_argument("--work_dir", type=str, default="work_dir", help="work directory")
    main(parser.parse_args())
