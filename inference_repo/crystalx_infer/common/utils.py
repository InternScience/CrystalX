import os
import re
from collections import Counter
from itertools import product

import numpy as np
from scipy.spatial.distance import cdist


def extract_non_numeric_prefix(string):
    match = re.match(r"^[a-zA-Z]+", string)
    return match.group() if match else ""


def check_sfac(sfac_lst, shelxt_pred):
    if "B" in sfac_lst and "B" not in shelxt_pred:
        shelxt_pred[-1] = "B"
    if "FE" in sfac_lst and "FE" not in shelxt_pred:
        elst = ["FE", "CO", "NI", "CU", "ZN"]
        for i in range(len(shelxt_pred)):
            if shelxt_pred[i] in elst and shelxt_pred[i + 1] not in elst:
                shelxt_pred[i] = "FE"
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
    with open(res_file_path, "r") as res_file:
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
        if is_atom_name:
            atom_name = first_line[0]
        else:
            atom_name = extract_non_numeric_prefix(first_line[0])
        shelxt_pred.append(atom_name)
        x = float(first_line[2])
        y = float(first_line[3])
        z = float(first_line[4])
        coord.append([x, y, z])
        qvalue_list.append(float(first_line[-1]))
        isotropy_list.append(float(first_line[-2]))
    if is_check_sfac:
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
    with open(res_file_path, "r") as res_file:
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
        if is_atom_name:
            atom_name = first_line[0]
        else:
            atom_name = extract_non_numeric_prefix(first_line[0])
        if not is_hydro and atom_name == "H":
            continue
        shelxt_pred.append(atom_name)
        x = float(first_line[2])
        y = float(first_line[3])
        z = float(first_line[4])
        coord.append([x, y, z])
    return coord, shelxt_pred


def update_shelxt(
    res_file_path,
    new_res_file_path,
    atom_list,
    no_sfac=False,
    refine_round=10,
):
    fname = os.path.basename(new_res_file_path)
    fname = os.path.splitext(fname)[0]
    sfac = {}
    p = 0
    tail_trimmed = 0
    atom_cnt = {}
    atom_list = [item.capitalize() for item in atom_list]
    if no_sfac:
        sfac_line = "SFAC "
        unit_line = "UNIT "
        pred_sfac = dict(Counter(atom_list))
        for k, v in pred_sfac.items():
            sfac_line += f"{k} "
            unit_line += f"{str(v)} "
        sfac_line += "H\n"
        unit_line += "37\n"
    with open(res_file_path, "r") as res_file, open(new_res_file_path, "w") as new_res_file:
        start_reading = False
        for line in res_file:
            if not start_reading:
                if "TITL" in line and line.split()[0] == "TITL":
                    line = f"TITL {fname}\n"
                if "SFAC" in line:
                    if no_sfac:
                        line = sfac_line
                    sfac_atom = line.split()[1:]
                    for i, atom in enumerate(sfac_atom):
                        sfac[atom.capitalize()] = i + 1
                if "UNIT" in line and no_sfac:
                    line = unit_line
                if "L.S." in line:
                    line = f"L.S. {str(refine_round)}\n"
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
                    if atom not in atom_cnt:
                        atom_cnt[atom] = 1
                    else:
                        atom_cnt[atom] += 1
                    atom_name = atom + str(atom_cnt[atom])
                    atom_pos[0] = atom_name
                    atom_pos[1] = str(sfac[atom])
                    atom_pos = " ".join(atom_pos)
                    atom_pos += "\n"
                    p += 1
                    new_res_file.write(atom_pos)
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
    if is_weight:
        with open(res_file_path, "r") as res_file:
            start_reading = False
            for line in res_file:
                if start_reading and "WGHT" in line:
                    weight_line = line
                if "END" in line:
                    start_reading = True
    with open(res_file_path, "r") as res_file, open(new_res_file_path, "w") as new_res_file:
        for line in res_file:
            if re_afix:
                if "AFIX" in line:
                    if atom_name == "N":
                        line = "AFIX 3\n"
                else:
                    line_info = line.split()
                    if len(line_info) > 0 and any(char.isalpha() for char in line_info[0]):
                        atom_name = extract_non_numeric_prefix(line_info[0])
            if "WGHT" in line and is_weight:
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
    with open(res_file_path, "r") as res_file, open(new_res_file_path, "w") as new_res_file:
        for line in res_file:
            if "ZERR" in line:
                if no_given_sfac:
                    zerr = 1.00
                    sp_line = line.split()
                    sp_line[1] = str(zerr)
                    line = " ".join(sp_line)
                    line += "\n"
                else:
                    zerr = float(line.split()[1])
            if "UNIT" in line:
                sp_line = line.split()
                if "H" in sfac.keys():
                    cell_hydro_num = int(sfac["H"] * zerr)
                    sp_line[-1] = str(cell_hydro_num)
                line = " ".join(sp_line)
                line += "\n"
            if "LIST 6" in line:
                line = "LIST 4\n"
            new_res_file.write(line)


def read_checkcif(chk_path):
    ln_max = 10
    is_structure_valid = True
    is_quality_valid = True
    with open(chk_path, "r") as chk_file:
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
    with open(lst_file_path, "r") as lst_file:
        start_reading = False
        for line in lst_file:
            if "connectivity table" in line:
                start_reading = True
                continue
            if start_reading:
                if "-" in line:
                    key = line.split()[0]
                    if "no bonds found" in line:
                        value = []
                    else:
                        value = line.split()[2:]
                    mol_graph[key] = value
                if "Reflections" in line or "Operators for generating equivalent atoms" in line:
                    break
    return mol_graph


def gen_hfix_ins(ins_file_path, mol_graph, hydro_num):
    atom_name = []
    try:
        _, atom_name = load_shelxt_final(
            ins_file_path,
            begin_flag="FVAR",
            is_atom_name=True,
            is_hydro=False,
        )
    except Exception:
        atom_name = []

    if len(atom_name) == 0:
        _, atom_name, _, _ = load_shelxt(ins_file_path, begin_flag="ANIS", is_atom_name=True)

    hydro_type = {
        "HFIX 13": [],
        "HFIX 23": [],
        "HFIX 137": [],
        "HFIX 153": [],
        "HFIX 163": [],
        "HFIX 43": [],
        "HFIX 93": [],
        "HFIX 147": [],
    }
    for atom, h in zip(atom_name, hydro_num):
        if hasattr(h, "item"):
            h = int(h.item())
        else:
            h = int(h)
        if h == 0 or h > 3:
            continue
        connect_atom = mol_graph.get(atom, [])
        atom_type = extract_non_numeric_prefix(atom)
        if atom_type == "C":
            if h == 1:
                if len(connect_atom) == 2:
                    hydro_type["HFIX 43"].append(atom)
                elif len(connect_atom) == 1:
                    hydro_type["HFIX 163"].append(atom)
                elif len(connect_atom) == 3:
                    hydro_type["HFIX 13"].append(atom)
            if h == 2:
                if len(connect_atom) == 1:
                    hydro_type["HFIX 93"].append(atom)
                elif len(connect_atom) == 2:
                    hydro_type["HFIX 23"].append(atom)
                elif len(connect_atom) == 3:
                    hydro_type["HFIX 13"].append(atom)
            if h == 3:
                if len(connect_atom) == 1:
                    hydro_type["HFIX 137"].append(atom)
                elif len(connect_atom) == 2:
                    hydro_type["HFIX 23"].append(atom)
                elif len(connect_atom) == 3:
                    hydro_type["HFIX 13"].append(atom)
        if atom_type == "N":
            if h == 1:
                if len(connect_atom) == 2:
                    hydro_type["HFIX 43"].append(atom)
                if len(connect_atom) == 3:
                    hydro_type["HFIX 13"].append(atom)
            if h == 2:
                if len(connect_atom) == 1:
                    hydro_type["HFIX 93"].append(atom)
                if len(connect_atom) == 2:
                    hydro_type["HFIX 23"].append(atom)
            if h == 3 and len(connect_atom) == 1:
                hydro_type["HFIX 137"].append(atom)
        if atom_type == "O":
            if h == 1 and len(connect_atom) == 1:
                hydro_type["HFIX 147"].append(atom)
        if atom_type == "B":
            if h == 1:
                hydro_type["HFIX 153"].append(atom)
        if atom_type not in ["C", "N", "O", "B"] and h == 1:
            hydro_type["HFIX 43"].append(atom)
    hfix_ins = []
    for key, value in hydro_type.items():
        if len(value) == 0:
            continue
        groups = [value[i : i + 10] for i in range(0, len(value), 10)]
        for item in groups:
            hfix_ins.append(key + " " + " ".join(item) + "\n")
    return hfix_ins


def update_shelxt_hydro(res_file_path, new_res_file_path, hfix_ins):
    with open(res_file_path, "r") as res_file, open(new_res_file_path, "w") as new_res_file:
        for line in res_file:
            if "BOND" in line:
                new_res_file.write("\n")
                for hfix in hfix_ins:
                    new_res_file.write(hfix)
                new_res_file.write("BOND $H\n")
                new_res_file.write("CONF\n")
                new_res_file.write("HTAB\n")
            elif "END" in line:
                new_res_file.write(line)
                break
            else:
                new_res_file.write(line)


def get_R2(res_file_path):
    with open(res_file_path, "r") as res_file:
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
    with open(res_file_path, "r") as res_file, open(new_res_file_path, "w") as new_res_file:
        for line in res_file:
            new_res_file.write(line)


def get_equiv_pos2(ideal_frac, gt, structure, radius=3):
    uc = structure.crystal_symmetry().unit_cell()
    ideal_cart = [list(uc.orthogonalize(point)) for point in ideal_frac]
    ideal_cart = np.array(ideal_cart)
    equiv_ideal_frac = []
    equiv_gt = []
    is_equiv = True
    for i in range(len(gt)):
        atom_site = ideal_frac[i]
        equiv_sites = structure.sym_equiv_sites(atom_site.tolist()).coordinates()
        equiv_sites = [list(site) for site in equiv_sites]
        equiv_ideal_frac += equiv_sites
        equiv_gt += [gt[i]] * len(equiv_sites)
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
