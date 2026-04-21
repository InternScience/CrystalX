"""SHELX parsing and editing helpers."""

from __future__ import annotations

import os
import re
from collections import Counter
from itertools import product

import numpy as np
from scipy.spatial.distance import cdist


def extract_non_numeric_prefix(string: str) -> str:
    match = re.match(r"^[a-zA-Z]+", string)
    return match.group() if match else ""


def check_sfac(sfac_lst, shelxt_pred):
    if "B" in sfac_lst and "B" not in shelxt_pred and len(shelxt_pred) > 0:
        shelxt_pred[-1] = "B"
    if "FE" in sfac_lst and "FE" not in shelxt_pred:
        metal_candidates = {"FE", "CO", "NI", "CU", "ZN"}
        for index, atom_name in enumerate(shelxt_pred):
            next_atom = shelxt_pred[index + 1] if index + 1 < len(shelxt_pred) else None
            if atom_name in metal_candidates and next_atom not in metal_candidates:
                shelxt_pred[index] = "FE"
                break
    return shelxt_pred


def load_shelxt(
    res_file_path,
    begin_flag="PLAN",
    end_flag="HKLF",
    is_atom_name=False,
    is_check_sfac=False,
):
    qpeak = []
    coord = []
    shelxt_pred = []
    qvalue_list = []
    isotropy_list = []
    sfac_lst = []

    with open(res_file_path, "r", encoding="utf-8", errors="ignore") as res_file:
        start_reading = False
        for line in res_file:
            if "SFAC" in line:
                sfac_lst = line.split()[1:]
            if start_reading:
                if end_flag in line:
                    break
                qpeak.append(line.split())
            if begin_flag in line:
                start_reading = True
                continue

    for first_line in qpeak:
        atom_name = first_line[0] if is_atom_name else extract_non_numeric_prefix(first_line[0])
        shelxt_pred.append(atom_name)
        coord.append([float(first_line[2]), float(first_line[3]), float(first_line[4])])
        qvalue_list.append(float(first_line[-1]))
        isotropy_list.append(float(first_line[-2]))

    if is_check_sfac and sfac_lst:
        shelxt_pred = check_sfac(sfac_lst, shelxt_pred)
    return coord, shelxt_pred, qvalue_list, isotropy_list


def load_shelxt_final(
    res_file_path,
    begin_flag="PLAN",
    end_flag="HKLF",
    is_atom_name=False,
    is_hydro=True,
):
    qpeak = []
    coord = []
    shelxt_pred = []

    with open(res_file_path, "r", encoding="utf-8", errors="ignore") as res_file:
        start_reading = False
        for line in res_file:
            if start_reading:
                if end_flag in line:
                    break
                if "AFIX" in line:
                    continue
                line_info = line.split()
                if len(line_info) == 0:
                    continue
                first_str = line_info[0]
                if any(char.isalpha() for char in first_str):
                    qpeak.append(line_info)
            if begin_flag in line:
                start_reading = True
                continue

    for first_line in qpeak:
        atom_name = first_line[0] if is_atom_name else extract_non_numeric_prefix(first_line[0])
        if not is_hydro and atom_name == "H":
            continue
        shelxt_pred.append(atom_name)
        coord.append([float(first_line[2]), float(first_line[3]), float(first_line[4])])
    return coord, shelxt_pred


def update_shelxt(
    res_file_path,
    new_res_file_path,
    atom_list,
    no_sfac=False,
    refine_round=10,
):
    fname = os.path.splitext(os.path.basename(new_res_file_path))[0]
    sfac = {}
    p = 0
    tail_trimmed = 0
    atom_cnt = {}
    atom_list = [item.capitalize() for item in atom_list]

    if no_sfac:
        sfac_line = "SFAC "
        unit_line = "UNIT "
        pred_sfac = dict(Counter(atom_list))
        for atom_symbol, count in pred_sfac.items():
            sfac_line += f"{atom_symbol} "
            unit_line += f"{count} "
        sfac_line += "H\n"
        unit_line += "37\n"

    with open(res_file_path, "r", encoding="utf-8", errors="ignore") as res_file, open(
        new_res_file_path, "w", encoding="utf-8"
    ) as new_res_file:
        start_reading = False
        for line in res_file:
            if not start_reading:
                if "TITL" in line and line.split()[0] == "TITL":
                    line = f"TITL {fname}\n"
                if "SFAC" in line:
                    if no_sfac:
                        line = sfac_line
                    sfac_atom = line.split()[1:]
                    for index, atom in enumerate(sfac_atom):
                        sfac[atom.capitalize()] = index + 1
                if "UNIT" in line and no_sfac:
                    line = unit_line
                if "L.S." in line:
                    line = f"L.S. {refine_round}\n"
                if "LIST" in line:
                    line = "LIST 4\n"
                new_res_file.write(line)

            if start_reading:
                if "HKLF" in line:
                    new_res_file.write(line)
                    start_reading = False
                else:
                    if p >= len(atom_list):
                        tail_trimmed += 1
                        continue
                    atom_pos = line.split()
                    atom = atom_list[p].capitalize()
                    atom_cnt[atom] = atom_cnt.get(atom, 0) + 1
                    atom_name = atom + str(atom_cnt[atom])
                    atom_pos[0] = atom_name
                    atom_pos[1] = str(sfac[atom])
                    new_res_file.write(" ".join(atom_pos) + "\n")
                    p += 1

            if "PLAN" in line:
                new_res_file.write("ANIS\n")
                start_reading = True

    if tail_trimmed > 0:
        print(
            f"[WARN] update_shelxt: trimmed tail atom rows={tail_trimmed} "
            f"(pred_n={len(atom_list)})"
        )


def update_shelxt_weight(
    res_file_path,
    new_res_file_path,
    is_weight=True,
    is_acta=True,
    re_afix=False,
):
    weight_line = ""
    atom_name = ""

    if is_weight:
        with open(res_file_path, "r", encoding="utf-8", errors="ignore") as res_file:
            start_reading = False
            for line in res_file:
                if start_reading and "WGHT" in line:
                    weight_line = line
                if "END" in line:
                    start_reading = True

    with open(res_file_path, "r", encoding="utf-8", errors="ignore") as res_file, open(
        new_res_file_path, "w", encoding="utf-8"
    ) as new_res_file:
        for line in res_file:
            if re_afix:
                if "AFIX" in line:
                    if atom_name == "N":
                        line = "AFIX 3\n"
                else:
                    line_info = line.split()
                    if len(line_info) > 0 and any(char.isalpha() for char in line_info[0]):
                        atom_name = extract_non_numeric_prefix(line_info[0])
            if "WGHT" in line and is_weight and weight_line:
                if is_acta:
                    new_res_file.write("ACTA\n")
                line = weight_line
            if "LIST 6" in line:
                line = "LIST 4\n"
            if "END" in line:
                new_res_file.write("END")
                break
            new_res_file.write(line)


def update_shelxt_final(res_file_path, new_res_file_path, sfac, no_given_sfac=False):
    with open(res_file_path, "r", encoding="utf-8", errors="ignore") as res_file, open(
        new_res_file_path, "w", encoding="utf-8"
    ) as new_res_file:
        for line in res_file:
            if "ZERR" in line:
                if no_given_sfac:
                    zerr = 1.00
                    sp_line = line.split()
                    sp_line[1] = str(zerr)
                    line = " ".join(sp_line) + "\n"
                else:
                    zerr = float(line.split()[1])
            if "UNIT" in line:
                sp_line = line.split()
                if "H" in sfac.keys():
                    cell_hydro_num = int(sfac["H"] * zerr)
                    sp_line[-1] = str(cell_hydro_num)
                line = " ".join(sp_line) + "\n"
            if "LIST 6" in line:
                line = "LIST 4\n"
            new_res_file.write(line)


def read_checkcif(chk_path):
    ln_max = 10
    is_structure_valid = True
    is_quality_valid = True
    with open(chk_path, "r", encoding="utf-8", errors="ignore") as chk_file:
        line_cnt = 0
        for line in chk_file:
            if "#=======================" in line:
                line_cnt += 1
            if line_cnt == ln_max:
                break
            if "ALERT_2_B" in line or "ALERT_2_A" in line:
                print(line)
                is_structure_valid = False
            if "ALERT_3_B" in line or "ALERT_3_A" in line:
                is_quality_valid = False
    return is_structure_valid, is_quality_valid


def get_bond(lst_file_path):
    mol_graph = {}
    with open(lst_file_path, "r", encoding="utf-8", errors="ignore") as lst_file:
        start_reading = False
        for line in lst_file:
            if "connectivity table" in line:
                start_reading = True
                continue
            if start_reading:
                if "-" in line:
                    key = line.split()[0]
                    value = [] if "no bonds found" in line else line.split()[2:]
                    mol_graph[key] = value
                if "Reflections" in line or "Operators for generating equivalent atoms" in line:
                    break
    return mol_graph


def update_shelxt_hydro(res_file_path, new_res_file_path, hfix_ins):
    inserted_hfix = False
    with open(res_file_path, "r", encoding="utf-8", errors="ignore") as res_file, open(
        new_res_file_path, "w", encoding="utf-8"
    ) as new_res_file:
        for line in res_file:
            if "BOND" in line:
                new_res_file.write("\n")
                for hfix in hfix_ins:
                    new_res_file.write(hfix)
                new_res_file.write("BOND $H\n")
                new_res_file.write("CONF\n")
                new_res_file.write("HTAB\n")
                inserted_hfix = True
            elif "END" in line:
                if not inserted_hfix:
                    new_res_file.write("\n")
                    for hfix in hfix_ins:
                        new_res_file.write(hfix)
                    new_res_file.write("BOND $H\n")
                    new_res_file.write("CONF\n")
                    new_res_file.write("HTAB\n")
                new_res_file.write(line)
                break
            else:
                new_res_file.write(line)


def get_R2(res_file_path):
    with open(res_file_path, "r", encoding="utf-8", errors="ignore") as res_file:
        start_reading = False
        for line in res_file:
            if "REM wR2" in line:
                start_reading = True
                wr2 = float(line.split()[3][:-1])
                goof = float(line.split()[8][:-1])
            if "REM R1" in line:
                r1 = float(line.split()[3])
            if start_reading and "Q1" in line.split():
                q1 = float(line.split()[7])
    return wr2, r1, goof, q1


def copy_file(res_file_path, new_res_file_path):
    with open(res_file_path, "r", encoding="utf-8", errors="ignore") as res_file, open(
        new_res_file_path, "w", encoding="utf-8"
    ) as new_res_file:
        for line in res_file:
            new_res_file.write(line)


def get_equiv_pos2(ideal_frac, gt, structure, radius=3):
    uc = structure.crystal_symmetry().unit_cell()
    ideal_cart = [list(uc.orthogonalize(point)) for point in ideal_frac]
    ideal_cart = np.array(ideal_cart)
    equiv_ideal_frac = []
    equiv_gt = []
    is_equiv = True
    for index in range(len(gt)):
        atom_site = ideal_frac[index]
        equiv_sites = structure.sym_equiv_sites(atom_site.tolist()).coordinates()
        equiv_sites = [list(site) for site in equiv_sites]
        equiv_ideal_frac += equiv_sites
        equiv_gt += [gt[index]] * len(equiv_sites)
    equiv_ideal_frac = np.array(equiv_ideal_frac)
    if equiv_ideal_frac.shape[0] < 1:
        is_equiv = False

    if is_equiv:
        expand_equiv_ideal_frac = []
        expand_equiv_gt = []
        translations = list(product([-2, -1, 0, 1, 2], repeat=3))
        for op in translations:
            expand_equiv_ideal_frac.append(equiv_ideal_frac + op)
            expand_equiv_gt += equiv_gt
        expand_equiv_ideal_frac = np.concatenate(expand_equiv_ideal_frac, axis=0)
        expand_equiv_gt = np.array(expand_equiv_gt)
        equiv_ideal_cart = [list(uc.orthogonalize(point)) for point in expand_equiv_ideal_frac]
        equiv_ideal_cart = np.array(equiv_ideal_cart)
        distances = cdist(ideal_cart, equiv_ideal_cart)
        min_dist = np.min(distances, axis=1)
        if np.min(min_dist) < radius:
            min_index = np.where(distances < radius)[1]
            min_index = [
                item
                for item in min_index
                if np.min(cdist(equiv_ideal_cart[[item]], ideal_cart)) > 0.2
            ]
            ideal_cart = np.concatenate([ideal_cart, equiv_ideal_cart[min_index]], axis=0)
            gt = gt + expand_equiv_gt[min_index].tolist()
    return ideal_cart, gt
