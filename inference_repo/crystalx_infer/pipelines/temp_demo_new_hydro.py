from rdkit import Chem
from iotbx.data_manager import DataManager    # Load in the DataManager
from iotbx.shelx import crystal_symmetry_from_ins
from cctbx.xray.structure import structure
import os
import json
from crystalx_infer.common.utils import *
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from crystalx_infer.models.torchmd_et import TorchMD_ET
from crystalx_infer.models.noise_output_model import (
    EquivariantScalar,
    EquivariantVectorOutput,
)
from crystalx_infer.models.torchmd_net import TorchMD_Net
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import argparse

HIDDEN_CHANNELS = 512


def _build_graph_from_ins(ins_file_path, radius_scale=1.20, min_dist=0.10):
    """
    Fallback connectivity graph when .lst from SHELXL is unavailable.
    Build bonds from cartesian distances using covalent radii.
    """
    frac = []
    atom_names = []
    try:
        frac_fvar, atom_names_fvar = load_shelxt_final(
            ins_file_path, begin_flag='FVAR', is_atom_name=True, is_hydro=False
        )
        if len(atom_names_fvar) > 0:
            frac = frac_fvar
            atom_names = atom_names_fvar
    except Exception:
        pass

    if len(atom_names) == 0:
        frac_anis, atom_names_anis, _, _ = load_shelxt(
            ins_file_path, begin_flag='ANIS', is_atom_name=True
        )
        frac = []
        atom_names = []
        for p, name in zip(frac_anis, atom_names_anis):
            atom_sym = extract_non_numeric_prefix(name).capitalize()
            if not atom_sym or atom_sym == 'H':
                continue
            frac.append(p)
            atom_names.append(name)

    filtered = []
    for p, name in zip(frac, atom_names):
        sym = extract_non_numeric_prefix(name).capitalize()
        if not sym or sym == 'H':
            continue
        filtered.append((p, name, sym))

    mol_graph = {name: [] for _, name, _ in filtered}
    if len(filtered) <= 1:
        return mol_graph

    sym = crystal_symmetry_from_ins.extract_from(ins_file_path)
    uc = sym.unit_cell()
    cart = np.array([list(uc.orthogonalize(p)) for p, _, _ in filtered], dtype=np.float64)

    pt = Chem.GetPeriodicTable()
    radii = []
    for _, _, atom_sym in filtered:
        try:
            z = Chem.Atom(atom_sym).GetAtomicNum()
            r = float(pt.GetRcovalent(int(z)))
            if r <= 0:
                r = 0.77
        except Exception:
            r = 0.77
        radii.append(r)

    n = len(filtered)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(cart[i] - cart[j]))
            cutoff = (radii[i] + radii[j]) * float(radius_scale)
            if d > float(min_dist) and d <= cutoff:
                ni = filtered[i][1]
                nj = filtered[j][1]
                mol_graph[ni].append(nj)
                mol_graph[nj].append(ni)

    return mol_graph


def _load_heavy_from_template(res_file_path, ins_file_path):
    def _load_from_anis(path):
        frac, atom_names, _, _ = load_shelxt(
            path, begin_flag='ANIS', is_atom_name=True
        )
        real_frac = []
        shelxt_pred = []
        for p, name in zip(frac, atom_names):
            atom_sym = extract_non_numeric_prefix(name).capitalize()
            if not atom_sym or atom_sym == 'H':
                continue
            real_frac.append(p)
            shelxt_pred.append(atom_sym)
        return np.array(real_frac), shelxt_pred

    # Preferred: refined/resolved heavy structure from .res.
    if os.path.exists(res_file_path):
        real_frac, shelxt_pred = load_shelxt_final(
            res_file_path, begin_flag='FVAR', is_hydro=False
        )
        if len(shelxt_pred) > 0:
            return np.array(real_frac), shelxt_pred, res_file_path

        # If FVAR section is absent (e.g. AI.res copied from AI.ins), fallback to ANIS parse.
        real_frac2, shelxt_pred2 = _load_from_anis(res_file_path)
        if len(shelxt_pred2) > 0:
            return real_frac2, shelxt_pred2, res_file_path

    # Fallback: parse heavy atoms directly from AI .ins (ANIS section).
    real_frac3, shelxt_pred3 = _load_from_anis(ins_file_path)
    return real_frac3, shelxt_pred3, ins_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hydro Handler')
    parser.add_argument('--fname', type=str, default='sample2_AI', help='file name')
    parser.add_argument('--work_dir', type=str, default='work_dir', help='work directory')
    parser.add_argument('--model_path', type=str, default='final_hydro_model_add_no_noise_fold_3.pth', help='model path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fname = args.fname
    work_dir = args.work_dir
    model_path = args.model_path

    ins_file_path = f'{work_dir}/{fname}.ins'
    hkl_file_path = f'{work_dir}/{fname}.hkl'
    # lst_file_path = f'{work_dir}/{fname}.lst'
    lst_file_path = f'{work_dir}/{fname}Bond.lst'
    res_file_path = f'{work_dir}/{fname}.res'
    new_res_file_path = f'{work_dir}/{fname}hydro.ins'
    new_hkl_file_path = f'{work_dir}/{fname}hydro.hkl'
    pred_summary_path = f'{work_dir}/{fname}_hydro_pred.json'

    real_frac, shelxt_pred, sym_src_path = _load_heavy_from_template(
        res_file_path=res_file_path,
        ins_file_path=ins_file_path,
    )

    # real_frac, shelxt_pred, _, _ = load_shelxt(ins_file_path, begin_flag = 'ANIS')
    real_num = len(shelxt_pred)
    print(shelxt_pred)

    sym = crystal_symmetry_from_ins.extract_from(sym_src_path)
    structure = structure(crystal_symmetry=sym)

    real_frac = np.array(real_frac)
    _real_cart, _z = get_equiv_pos2(real_frac, shelxt_pred, structure, radius=3.2)

    _z = [item.capitalize() for item in _z]
    _z = [Chem.Atom(item).GetAtomicNum() for item in _z]

    if _real_cart.shape[0] < real_num:
        raise ValueError(
            f"Expanded positions shorter than heavy atom count: "
            f"{_real_cart.shape[0]} < {real_num}"
        )

    # Keep the first real_num heavy atoms strictly aligned with template order.
    real_cart = [row.tolist() for row in _real_cart[:real_num]]
    z = [int(v) for v in _z[:real_num]]

    # Add unique symmetry-expanded neighbors after the main heavy atoms.
    for i in range(real_num, _real_cart.shape[0]):
        v = _real_cart[i].tolist()
        if v not in real_cart:
            real_cart.append(v)
            z.append(int(_z[i]))

    # Alignment check: template fractional -> cart should match first real_num rows.
    uc = structure.crystal_symmetry().unit_cell()
    ref_main_cart = np.array([list(uc.orthogonalize(p)) for p in real_frac], dtype=np.float64)
    used_main_cart = np.array(real_cart[:real_num], dtype=np.float64)
    main_align_max_abs = float(np.max(np.abs(ref_main_cart - used_main_cart)))
    main_align_ok = bool(main_align_max_abs < 1e-8)

    mask = np.array([0] * len(z))
    mask[:len(shelxt_pred)] = 1
    mask = torch.from_numpy(mask).bool()

    z = torch.tensor(z)
    real_cart = np.array(real_cart)
    real_cart = torch.from_numpy(real_cart.astype(np.float32))

    test_dataset = []
    test_dataset.append(Data(z = z, pos = real_cart, mask = mask))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    representation_model = TorchMD_ET(
        hidden_channels=HIDDEN_CHANNELS,
        attn_activation='silu',
        num_heads=8,
        distance_influence='both',
        )
    num_classes = 8
    output_model = EquivariantScalar(HIDDEN_CHANNELS, num_classes=num_classes)
    model = TorchMD_Net(representation_model=representation_model,
                        output_model=output_model)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.inference_mode():
        for data in tqdm(test_loader):
            data = data.to(device)
            logits = model(data.z, data.pos, data.batch)

            logits = logits[data.mask]
            logits = F.softmax(logits, dim = -1)

            _, predicted = torch.max(logits, dim=1)
            print(predicted)
            predicted = predicted.to('cpu')

    pred_h_per_atom = [int(x) for x in predicted.tolist()]
    pred_h_total = int(sum(pred_h_per_atom))


    lst_candidates = [lst_file_path, f'{work_dir}/{fname}.lst']
    mol_graph = None
    used_lst = ""
    for lp in lst_candidates:
        if os.path.exists(lp):
            mol_graph = get_bond(lp)
            used_lst = lp
            break

    if mol_graph is None:
        print("[WARN] No .lst found for bond table, fallback to distance-based connectivity.")
        # Prefer the same template used for hydro writing (usually *_AI.res).
        template_for_graph = res_file_path if os.path.exists(res_file_path) else ins_file_path
        mol_graph = _build_graph_from_ins(template_for_graph)
    else:
        print(f"[INFO] Bond table from: {used_lst}")

    template_path = res_file_path if os.path.exists(res_file_path) else ins_file_path

    # Ensure all heavy atom names in template file exist in graph dict keys.
    atom_names_in_template = []
    try:
        _, atom_names_in_template = load_shelxt_final(
            template_path, begin_flag='FVAR', is_atom_name=True, is_hydro=False
        )
    except Exception:
        atom_names_in_template = []
    if len(atom_names_in_template) == 0:
        frac_tmp, name_tmp, _, _ = load_shelxt(
            template_path, begin_flag='ANIS', is_atom_name=True
        )
        atom_names_in_template = []
        for _, an in zip(frac_tmp, name_tmp):
            if extract_non_numeric_prefix(an).upper() != 'H':
                atom_names_in_template.append(an)
    for atom_name in atom_names_in_template:
        mol_graph.setdefault(atom_name, [])

    if len(atom_names_in_template) != len(pred_h_per_atom):
        raise ValueError(
            f"Hydro prediction length mismatch: "
            f"template_heavy_atoms={len(atom_names_in_template)} "
            f"vs predicted={len(pred_h_per_atom)} "
            f"(template={template_path})"
        )

    copy_file(hkl_file_path, new_hkl_file_path)
    hfix_ins = gen_hfix_ins(template_path, mol_graph, predicted)
    update_shelxt_hydro(template_path, new_res_file_path, hfix_ins)

    hfix_hcount = {
        13: 1,
        23: 2,
        43: 1,
        93: 2,
        137: 3,
        147: 1,
        153: 1,
        163: 1,
    }
    hfix_total_h = 0
    for line in hfix_ins:
        parts = line.strip().split()
        if len(parts) < 3 or parts[0] != 'HFIX':
            continue
        try:
            code = int(parts[1])
        except Exception:
            continue
        atom_count = len(parts) - 2
        hfix_total_h += int(hfix_hcount.get(code, 0)) * int(atom_count)

    pred_summary = {
        "fname": fname,
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
    with open(pred_summary_path, "w", encoding="utf-8") as f:
        json.dump(pred_summary, f)
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
