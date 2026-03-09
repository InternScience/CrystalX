import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from cctbx.xray.structure import structure
from iotbx.shelx import crystal_symmetry_from_ins
from rdkit import Chem
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from crystalx_infer.common.utils import (
    copy_file,
    extract_non_numeric_prefix,
    gen_hfix_ins,
    get_bond,
    get_equiv_pos2,
    load_shelxt,
    load_shelxt_final,
    update_shelxt_hydro,
)
from crystalx_infer.models.noise_output_model import EquivariantScalar
from crystalx_infer.models.torchmd_et import TorchMD_ET
from crystalx_infer.models.torchmd_net import TorchMD_Net


HIDDEN_CHANNELS = 512
NUM_CLASSES = 8
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


def build_model(model_path, device):
    representation_model = TorchMD_ET(
        hidden_channels=HIDDEN_CHANNELS,
        attn_activation="silu",
        num_heads=8,
        distance_influence="both",
    )
    output_model = EquivariantScalar(HIDDEN_CHANNELS, num_classes=NUM_CLASSES)
    model = TorchMD_Net(
        representation_model=representation_model,
        output_model=output_model,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def build_graph_from_ins(ins_file_path, radius_scale=1.20, min_dist=0.10):
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
        pass

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
    cart = np.array([list(unit_cell.orthogonalize(point)) for point, _, _ in filtered], dtype=np.float64)

    periodic_table = Chem.GetPeriodicTable()
    radii = []
    for _, _, atom_symbol in filtered:
        try:
            atomic_num = Chem.Atom(atom_symbol).GetAtomicNum()
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
    expanded_z = [Chem.Atom(symbol.capitalize()).GetAtomicNum() for symbol in expanded_symbols]

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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ins_file_path = f"{args.work_dir}/{args.fname}.ins"
    hkl_file_path = f"{args.work_dir}/{args.fname}.hkl"
    lst_file_path = f"{args.work_dir}/{args.fname}Bond.lst"
    res_file_path = f"{args.work_dir}/{args.fname}.res"
    new_res_file_path = f"{args.work_dir}/{args.fname}hydro.ins"
    new_hkl_file_path = f"{args.work_dir}/{args.fname}hydro.hkl"
    pred_summary_path = f"{args.work_dir}/{args.fname}_hydro_pred.json"

    real_frac, shelxt_pred, sym_src_path = load_heavy_from_template(
        res_file_path=res_file_path,
        ins_file_path=ins_file_path,
    )
    print(shelxt_pred)

    crystal_symmetry = crystal_symmetry_from_ins.extract_from(sym_src_path)
    crystal_structure = structure(crystal_symmetry=crystal_symmetry)

    real_num, real_cart, z = build_hydro_inputs(real_frac, shelxt_pred, crystal_structure)
    main_align_ok, main_align_max_abs = compute_alignment_metrics(real_frac, real_cart, crystal_structure)

    mask = torch.zeros(len(z), dtype=torch.bool)
    mask[: len(shelxt_pred)] = True

    test_loader = DataLoader(
        [Data(z=torch.tensor(z), pos=torch.from_numpy(np.array(real_cart, dtype=np.float32)), mask=mask)],
        batch_size=1,
        shuffle=False,
    )
    model = build_model(args.model_path, device)

    predicted = None
    with torch.inference_mode():
        for data in tqdm(test_loader):
            data = data.to(device)
            logits = model(data.z, data.pos, data.batch)
            logits = F.softmax(logits[data.mask], dim=-1)
            predicted = torch.argmax(logits, dim=1).to("cpu")
            print(predicted)

    pred_h_per_atom = [int(value) for value in predicted.tolist()]
    pred_h_total = int(sum(pred_h_per_atom))

    mol_graph = None
    used_lst = ""
    for candidate in [lst_file_path, f"{args.work_dir}/{args.fname}.lst"]:
        if os.path.exists(candidate):
            mol_graph = get_bond(candidate)
            used_lst = candidate
            break

    if mol_graph is None:
        print("[WARN] No .lst found for bond table, fallback to distance-based connectivity.")
        template_for_graph = res_file_path if os.path.exists(res_file_path) else ins_file_path
        mol_graph = build_graph_from_ins(template_for_graph)
    else:
        print(f"[INFO] Bond table from: {used_lst}")

    template_path = res_file_path if os.path.exists(res_file_path) else ins_file_path
    atom_names_in_template = load_template_atom_names(template_path)
    for atom_name in atom_names_in_template:
        mol_graph.setdefault(atom_name, [])

    if len(atom_names_in_template) != len(pred_h_per_atom):
        raise ValueError(
            "Hydro prediction length mismatch: "
            f"template_heavy_atoms={len(atom_names_in_template)} "
            f"vs predicted={len(pred_h_per_atom)} "
            f"(template={template_path})"
        )

    copy_file(hkl_file_path, new_hkl_file_path)
    hfix_ins = gen_hfix_ins(template_path, mol_graph, predicted)
    update_shelxt_hydro(template_path, new_res_file_path, hfix_ins)

    hfix_total_h = build_hfix_summary(hfix_ins)
    pred_summary = {
        "fname": args.fname,
        "pred_h_total": pred_h_total,
        "pred_h_per_atom": pred_h_per_atom,
        "pred_atom_count": int(len(pred_h_per_atom)),
        "heavy_atom_count_template": int(real_num),
        "coord_align_ok": bool(main_align_ok),
        "coord_align_max_abs_diff": float(main_align_max_abs),
        "hfix_line_count": int(len(hfix_ins)),
        "hfix_total_h": int(hfix_total_h),
        "pred_vs_hfix_match": bool(int(pred_h_total) == int(hfix_total_h)),
    }
    with open(pred_summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(pred_summary, file_obj)

    print(f"[INFO] Saved hydro prediction summary: {pred_summary_path}")
    print(f"[INFO] Predicted total H: {pred_h_total}")
    print(
        f"[INFO] HFIX implied total H: {pred_summary['hfix_total_h']} | "
        f"pred_vs_hfix_match={pred_summary['pred_vs_hfix_match']}"
    )
    print(
        f"[INFO] Coord align check: ok={pred_summary['coord_align_ok']} "
        f"max_abs_diff={pred_summary['coord_align_max_abs_diff']:.3e}"
    )
    print(predicted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydro Handler")
    parser.add_argument("--fname", type=str, default="sample2_AI", help="file name")
    parser.add_argument("--work_dir", type=str, default="work_dir", help="work directory")
    parser.add_argument(
        "--model_path",
        type=str,
        default="final_hydro_model_add_no_noise_fold_3.pth",
        help="model path",
    )
    main(parser.parse_args())
