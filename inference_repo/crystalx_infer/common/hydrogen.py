"""Hydrogen inference, HFIX, and mol-graph helpers."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from crystalx_infer.common.chem import atomic_num_from_symbol, atomic_symbol_from_z
from crystalx_infer.common.shelx import (
    extract_non_numeric_prefix,
    get_equiv_pos2,
    load_shelxt,
    load_shelxt_final,
)


HFIX_HCOUNT = {
    13: 1,
    23: 2,
    43: 1,
    93: 2,
    137: 3,
    147: 1,
    153: 1,
    163: 1,
}


def build_graph_from_ins(ins_file_path, radius_scale=1.20, min_dist=0.10):
    from iotbx.shelx import crystal_symmetry_from_ins

    frac = []
    atom_names = []

    try:
        frac_fvar, atom_names_fvar = load_shelxt_final(
            ins_file_path,
            begin_flag="FVAR",
            is_atom_name=True,
            is_hydro=False,
        )
        if len(atom_names_fvar) > 0:
            frac = frac_fvar
            atom_names = atom_names_fvar
    except Exception:
        frac = []
        atom_names = []

    if len(atom_names) == 0:
        frac_anis, atom_names_anis, _, _ = load_shelxt(
            ins_file_path,
            begin_flag="ANIS",
            is_atom_name=True,
        )
        for point, atom_name in zip(frac_anis, atom_names_anis):
            atom_symbol = extract_non_numeric_prefix(atom_name).capitalize()
            if not atom_symbol or atom_symbol == "H":
                continue
            frac.append(point)
            atom_names.append(atom_name)

    filtered = []
    for point, atom_name in zip(frac, atom_names):
        atom_symbol = extract_non_numeric_prefix(atom_name).capitalize()
        if not atom_symbol or atom_symbol == "H":
            continue
        filtered.append((point, atom_name, atom_symbol))

    mol_graph = {atom_name: [] for _, atom_name, _ in filtered}
    if len(filtered) <= 1:
        return mol_graph

    crystal_symmetry = crystal_symmetry_from_ins.extract_from(ins_file_path)
    unit_cell = crystal_symmetry.unit_cell()
    cart = np.array(
        [list(unit_cell.orthogonalize(point)) for point, _, _ in filtered],
        dtype=np.float64,
    )

    from rdkit import Chem

    periodic_table = Chem.GetPeriodicTable()
    radii = []
    for _, _, atom_symbol in filtered:
        try:
            atomic_num = atomic_num_from_symbol(atom_symbol)
            radius = float(periodic_table.GetRcovalent(int(atomic_num)))
            if radius <= 0:
                radius = 0.77
        except Exception:
            radius = 0.77
        radii.append(radius)

    for idx_i in range(len(filtered)):
        for idx_j in range(idx_i + 1, len(filtered)):
            distance = float(np.linalg.norm(cart[idx_i] - cart[idx_j]))
            cutoff = (radii[idx_i] + radii[idx_j]) * float(radius_scale)
            if distance > float(min_dist) and distance <= cutoff:
                atom_i = filtered[idx_i][1]
                atom_j = filtered[idx_j][1]
                mol_graph[atom_i].append(atom_j)
                mol_graph[atom_j].append(atom_i)

    return mol_graph


def load_heavy_from_template(res_file_path, ins_file_path):
    def load_from_anis(path):
        frac, atom_names, _, _ = load_shelxt(path, begin_flag="ANIS", is_atom_name=True)
        real_frac = []
        shelxt_pred = []
        for point, atom_name in zip(frac, atom_names):
            atom_symbol = extract_non_numeric_prefix(atom_name).capitalize()
            if not atom_symbol or atom_symbol == "H":
                continue
            real_frac.append(point)
            shelxt_pred.append(atom_symbol)
        return np.array(real_frac), shelxt_pred

    if os.path.exists(res_file_path):
        real_frac, shelxt_pred = load_shelxt_final(
            res_file_path,
            begin_flag="FVAR",
            is_hydro=False,
        )
        if len(shelxt_pred) > 0:
            return np.array(real_frac), shelxt_pred, res_file_path

        real_frac, shelxt_pred = load_from_anis(res_file_path)
        if len(shelxt_pred) > 0:
            return real_frac, shelxt_pred, res_file_path

    real_frac, shelxt_pred = load_from_anis(ins_file_path)
    return real_frac, shelxt_pred, ins_file_path


def build_hydro_inputs(real_frac, shelxt_pred, crystal_structure):
    real_num = len(shelxt_pred)
    expanded_cart, expanded_symbols = get_equiv_pos2(
        np.array(real_frac),
        shelxt_pred,
        crystal_structure,
        radius=3.2,
    )
    expanded_z = [atomic_num_from_symbol(symbol.capitalize()) for symbol in expanded_symbols]

    if expanded_cart.shape[0] < real_num:
        raise ValueError(
            f"Expanded positions shorter than heavy atom count: {expanded_cart.shape[0]} < {real_num}"
        )

    real_cart = [row.tolist() for row in expanded_cart[:real_num]]
    z = [int(value) for value in expanded_z[:real_num]]
    for idx in range(real_num, expanded_cart.shape[0]):
        position = expanded_cart[idx].tolist()
        if position not in real_cart:
            real_cart.append(position)
            z.append(int(expanded_z[idx]))

    return real_num, real_cart, z


def compute_alignment_metrics(real_frac, real_cart, crystal_structure):
    unit_cell = crystal_structure.crystal_symmetry().unit_cell()
    ref_main_cart = np.array([list(unit_cell.orthogonalize(point)) for point in real_frac], dtype=np.float64)
    used_main_cart = np.array(real_cart[: len(real_frac)], dtype=np.float64)
    max_abs = float(np.max(np.abs(ref_main_cart - used_main_cart)))
    return bool(max_abs < 1e-8), max_abs


def load_template_atom_names(template_path):
    try:
        _, atom_names = load_shelxt_final(
            template_path,
            begin_flag="FVAR",
            is_atom_name=True,
            is_hydro=False,
        )
        if len(atom_names) > 0:
            return atom_names
    except Exception:
        pass

    frac_tmp, atom_names_tmp, _, _ = load_shelxt(
        template_path,
        begin_flag="ANIS",
        is_atom_name=True,
    )
    filtered_atom_names = []
    for _, atom_name in zip(frac_tmp, atom_names_tmp):
        if extract_non_numeric_prefix(atom_name).upper() != "H":
            filtered_atom_names.append(atom_name)
    return filtered_atom_names


def gen_hfix_ins(ins_file_path, mol_graph, hydro_num):
    atom_name = load_template_atom_names(ins_file_path)

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
        h = int(h.item()) if hasattr(h, "item") else int(h)
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


def build_hfix_summary(hfix_ins):
    hfix_total_h = 0
    for line in hfix_ins:
        parts = line.strip().split()
        if len(parts) < 3 or parts[0] != "HFIX":
            continue
        try:
            code = int(parts[1])
        except Exception:
            continue
        hfix_total_h += int(HFIX_HCOUNT.get(code, 0)) * int(len(parts) - 2)
    return hfix_total_h


def build_graph_from_equiv_ase(
    main_z_mask,
    pos_mask,
    ase_cutoff_mult=1.10,
    ase_extra_cutoff=0.00,
    ase_skin=0.00,
):
    from ase import Atoms
    from ase.neighborlist import NeighborList, natural_cutoffs

    z_list = [int(x) for x in main_z_mask.detach().cpu().tolist()]
    atom_symbols = [atomic_symbol_from_z(z) for z in z_list]
    if len(z_list) == 0:
        return atom_symbols, []

    pos = pos_mask.detach().cpu().numpy()
    if pos.ndim != 2 or pos.shape[0] != len(z_list):
        raise ValueError(f"Invalid pos shape for graph build: {tuple(pos.shape)}")

    atoms = Atoms(symbols=atom_symbols, positions=pos)
    cutoffs = natural_cutoffs(atoms, mult=ase_cutoff_mult)
    cutoffs = [float(cutoff) + float(ase_extra_cutoff) for cutoff in cutoffs]
    neighbor_list = NeighborList(
        cutoffs=cutoffs,
        self_interaction=False,
        bothways=True,
        skin=ase_skin,
    )
    neighbor_list.update(atoms)

    degree = []
    for idx in range(len(z_list)):
        neigh_idx, _ = neighbor_list.get_neighbors(idx)
        degree.append(int(len(neigh_idx)))
    return atom_symbols, degree


def build_graph_from_equiv_rdkit(main_z_mask, pos_mask, rdkit_cov_factor=1.30):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D

    z_list = [int(x) for x in main_z_mask.detach().cpu().tolist()]
    atom_symbols = [atomic_symbol_from_z(z) for z in z_list]
    if len(z_list) == 0:
        return atom_symbols, []

    pos = pos_mask.detach().cpu().numpy()
    if pos.ndim != 2 or pos.shape[0] != len(z_list):
        raise ValueError(f"Invalid pos shape for graph build: {tuple(pos.shape)}")

    rw_mol = Chem.RWMol()
    for atomic_num in z_list:
        rw_mol.AddAtom(Chem.Atom(int(atomic_num)))
    mol = rw_mol.GetMol()

    conformer = Chem.Conformer(len(z_list))
    for idx, (x_coord, y_coord, z_coord) in enumerate(pos.tolist()):
        conformer.SetAtomPosition(idx, Point3D(float(x_coord), float(y_coord), float(z_coord)))
    mol.AddConformer(conformer, assignId=True)

    try:
        rdDetermineBonds.DetermineConnectivity(mol, covFactor=float(rdkit_cov_factor))
    except TypeError:
        rdDetermineBonds.DetermineConnectivity(mol)

    degree = [int(mol.GetAtomWithIdx(idx).GetDegree()) for idx in range(len(z_list))]
    return atom_symbols, degree


def is_reasonable_hydrogen(atom_symbol, degree, hydro_count):
    h = int(hydro_count)
    if h < 0:
        return False
    if atom_symbol == "O":
        if h == 0:
            return True
        if h in {1, 2}:
            return degree <= 2
        if h == 3:
            return degree == 0
        return False
    if atom_symbol == "N":
        if h == 0:
            return True
        if h == 1:
            return degree <= 3
        if h == 2:
            return degree <= 2
        if h == 3:
            return degree <= 1
        if h == 4:
            return degree == 0
        return False
    if atom_symbol == "C":
        return 0 <= h <= max(0, 4 - degree)
    if atom_symbol == "B":
        return 0 <= h <= max(0, 3 - degree)
    return True


def adjust_prediction_by_mol_graph(prob, predicted, atom_symbols, degree_list):
    sorted_indices = torch.argsort(prob, dim=1, descending=True)
    adjusted = predicted.clone()

    changed = 0
    valid_checked = 0
    atom_n = min(len(atom_symbols), len(degree_list), adjusted.shape[0])
    for idx in range(atom_n):
        atom_symbol = atom_symbols[idx]
        degree = int(degree_list[idx])

        chosen = int(adjusted[idx].item())
        for candidate in sorted_indices[idx]:
            hydro_count = int(candidate.item())
            if is_reasonable_hydrogen(atom_symbol, degree, hydro_count):
                chosen = hydro_count
                break
        if chosen != int(adjusted[idx].item()):
            changed += 1
        adjusted[idx] = chosen
        valid_checked += 1

    return adjusted, changed, valid_checked


@torch.no_grad()
def predict_hydrogen_counts(model, input_z, pos, batch, mask):
    logits = model(input_z, pos, batch)
    prob = F.softmax(logits[mask], dim=-1)
    predicted = torch.argmax(prob, dim=1)
    return prob, predicted


def hydro_group_name(atomic_num: int, degree: int, hydro_count: int) -> str:
    if int(atomic_num) == 0 and int(hydro_count) == 0:
        return "NoH"
    element_symbol = atomic_symbol_from_z(atomic_num)
    if int(hydro_count) <= 0:
        return element_symbol
    if int(hydro_count) == 1:
        return f"{element_symbol}{int(degree)}H"
    return f"{element_symbol}{int(degree)}H{int(hydro_count)}"


def hydro_group_key(atomic_num: int, degree: int, hydro_count: int):
    if int(hydro_count) <= 0:
        return (0, 0, 0)
    return (int(atomic_num), int(degree), int(hydro_count))
