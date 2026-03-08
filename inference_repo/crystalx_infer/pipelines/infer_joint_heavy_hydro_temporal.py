import os
import argparse
import random
import itertools
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from rdkit import Chem
from scipy.spatial.distance import cdist

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from crystalx_infer.models.torchmd_et import TorchMD_ET
from crystalx_infer.models.noise_output_model import EquivariantScalar
from crystalx_infer.models.torchmd_net import TorchMD_Net


def set_seed(seed: int = 42, deterministic_cudnn: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_run_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def split_by_year_txt(
    txt_path: str,
    pt_dir: str,
    test_years=(2022, 2023, 2024),
    pt_prefix="equiv_",
    pt_suffix=".pt",
    strict=False,
):
    test_years = set(str(y) for y in test_years)

    train_files, test_files = [], []
    missing = []
    seen = set()

    with open(txt_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"[WARN] txt line {ln} format error: {line}")
                continue

            year = parts[0]
            cif_name = parts[-1]
            cif_stem = os.path.splitext(os.path.basename(cif_name))[0]
            pt_path = os.path.join(pt_dir, f"{pt_prefix}{cif_stem}{pt_suffix}")

            if pt_path in seen:
                continue
            seen.add(pt_path)

            if not os.path.exists(pt_path):
                missing.append(pt_path)
                continue

            if year in test_years:
                test_files.append(pt_path)
            else:
                train_files.append(pt_path)

    if not strict:
        all_pt = [
            os.path.join(pt_dir, fn)
            for fn in os.listdir(pt_dir)
            if fn.endswith(pt_suffix)
        ]
        test_set = set(test_files)
        listed_set = set(train_files) | test_set
        extra = [p for p in all_pt if p not in listed_set and p not in test_set]
        train_files += extra

    return train_files, test_files, missing


def infer_num_classes_from_state_dict(state_dict):
    key = "output_model.output_network.1.update_net.2.bias"
    if key in state_dict and state_dict[key].ndim == 1:
        dim = int(state_dict[key].shape[0])
        if dim % 2 == 0 and dim > 0:
            return dim // 2
    key = "output_model.output_network.1.update_net.2.weight"
    if key in state_dict and state_dict[key].ndim == 2:
        dim = int(state_dict[key].shape[0])
        if dim % 2 == 0 and dim > 0:
            return dim // 2
    key = "output_model.output_network.1.vec2_proj.weight"
    if key in state_dict and state_dict[key].ndim == 2:
        return int(state_dict[key].shape[0])
    raise ValueError("Cannot infer num_classes from checkpoint. Please pass explicit num_classes.")


def _atomic_num_list(symbol_list):
    return [Chem.Atom(item.capitalize()).GetAtomicNum() for item in symbol_list]


def _stem_from_pt_path(pt_path: str):
    stem = os.path.splitext(os.path.basename(pt_path))[0]
    return stem[6:] if stem.startswith("equiv_") else stem


def _elem_symbol(atomic_num: int) -> str:
    try:
        return Chem.GetPeriodicTable().GetElementSymbol(int(atomic_num))
    except Exception:
        return f"Z{int(atomic_num)}"


def _hydro_group_name(atomic_num: int, degree: int, hydro_count: int) -> str:
    if int(atomic_num) == 0 and int(hydro_count) == 0:
        return "NoH"
    elem = _elem_symbol(atomic_num)
    d = int(degree)
    h = int(hydro_count)
    if h <= 0:
        return elem
    if h == 1:
        return f"{elem}{d}H"
    return f"{elem}{d}H{h}"


def _hydro_group_key(atomic_num: int, degree: int, hydro_count: int):
    """
    Group rule for hydro metrics:
    - Merge all H=0 into one class: (0, 0, 0)
    - Keep H>0 split by heavy element and RDKit degree: (Z, degree, H)
    """
    z = int(atomic_num)
    d = int(degree)
    h = int(hydro_count)
    if h <= 0:
        return (0, 0, 0)
    return (z, d, h)


def load_excluded_stems(txt_path: str) -> set:
    stems = set()
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            base = os.path.basename(s)
            stem = os.path.splitext(base)[0]
            if stem.startswith("equiv_"):
                stem = stem[6:]
            stems.add(stem)
    return stems


def build_joint_dataset(file_list, heavy_noise_max=0.1, is_check_dist=True, dist_min=0.1):
    dataset = []
    stats = {
        "kept": 0,
        "drop_missing_keys": 0,
        "drop_noise": 0,
        "drop_element": 0,
        "drop_dist": 0,
        "drop_shape": 0,
        "drop_c_h_gt3": 0,
    }
    required = {"z", "equiv_gt", "gt", "hydro_gt", "pos", "noise_list"}

    for fname in tqdm(file_list, desc="Build joint dataset"):
        try:
            mol = torch.load(fname)
        except Exception:
            stats["drop_missing_keys"] += 1
            continue

        if not required.issubset(set(mol.keys())):
            stats["drop_missing_keys"] += 1
            continue

        noise = mol["noise_list"]
        if np.max(np.abs(noise)) > heavy_noise_max:
            stats["drop_noise"] += 1
            continue

        try:
            z_heavy = _atomic_num_list(mol["z"])
            z_equiv = _atomic_num_list(mol["equiv_gt"])
            gt_main = _atomic_num_list(mol["gt"])
        except Exception:
            stats["drop_element"] += 1
            continue

        hydro_gt = mol["hydro_gt"]
        if isinstance(hydro_gt, torch.Tensor):
            hydro_gt = hydro_gt.detach().cpu().tolist()
        elif isinstance(hydro_gt, np.ndarray):
            hydro_gt = hydro_gt.tolist()
        hydro_gt = [int(h) for h in hydro_gt]

        # Skip sample if any Carbon has hydro_gt > 3.
        pair_n = min(len(gt_main), len(hydro_gt))
        has_c_h_gt3 = False
        for i in range(pair_n):
            if gt_main[i] == 6 and hydro_gt[i] > 3:
                has_c_h_gt3 = True
                break
        if has_c_h_gt3:
            stats["drop_c_h_gt3"] += 1
            continue

        pos_arr = mol["pos"]

        real_cart = []
        keep_idx = []
        for i in range(pos_arr.shape[0]):
            p = pos_arr[i].tolist()
            if p not in real_cart:
                real_cart.append(p)
                keep_idx.append(i)
        real_cart = np.array(real_cart, dtype=np.float32)
        z_heavy_dedup = [z_heavy[i] for i in keep_idx]
        z_equiv_dedup = [z_equiv[i] for i in keep_idx]

        if is_check_dist:
            d = cdist(real_cart, real_cart)
            d = d + 10 * np.eye(d.shape[0])
            if np.min(d) < dist_min:
                stats["drop_dist"] += 1
                continue

        n = len(z_heavy_dedup)
        heavy_n = len(gt_main)
        hydro_n = len(hydro_gt)
        if (
            heavy_n > n
            or hydro_n > n
            or hydro_n > heavy_n
            or hydro_n <= 0
            or heavy_n <= 0
        ):
            stats["drop_shape"] += 1
            continue

        heavy_init_z = list(sorted(gt_main, reverse=True)) + z_heavy_dedup[heavy_n:]
        heavy_mask = np.zeros(n, dtype=np.int64)
        heavy_mask[:heavy_n] = 1
        hydro_mask = np.zeros(n, dtype=np.int64)
        hydro_mask[:hydro_n] = 1

        # Keep hydro/main label lengths aligned with hydro mask to avoid shape mismatch.
        main_gt_hydro = gt_main[:hydro_n]

        dataset.append(
            Data(
                heavy_init_z=torch.tensor(heavy_init_z, dtype=torch.long),
                heavy_raw_z=torch.tensor(z_heavy_dedup, dtype=torch.long),
                heavy_y=torch.tensor(gt_main, dtype=torch.long),
                heavy_mask=torch.from_numpy(heavy_mask).bool(),
                hydro_input_z=torch.tensor(z_equiv_dedup, dtype=torch.long),
                hydro_y=torch.tensor(hydro_gt, dtype=torch.long),
                hydro_mask=torch.from_numpy(hydro_mask).bool(),
                main_z=torch.tensor(z_equiv_dedup, dtype=torch.long),
                main_gt_hydro=torch.tensor(main_gt_hydro, dtype=torch.long),
                pos=torch.from_numpy(real_cart),
                fname=fname,
            )
        )
        stats["kept"] += 1

    print("[Build]", " ".join([f"{k}={v}" for k, v in stats.items()]))
    return dataset, stats


@torch.no_grad()
def enforce_coverage_by_prob(prob, sfac, pred):
    device = pred.device
    n, k = prob.shape
    if n == 0 or k == 0 or n < k:
        return pred

    elem2k = {int(sfac[j].item()): j for j in range(k)}
    pred_k = torch.tensor([elem2k[int(x.item())] for x in pred], device=device, dtype=torch.long)
    counts = torch.bincount(pred_k, minlength=k).clone()
    used_atoms = set()
    missing = (counts == 0).nonzero(as_tuple=False).view(-1).tolist()

    while missing:
        changed_any = False
        for miss_k in missing:
            safe_mask = counts[pred_k] > 1
            if used_atoms:
                used = torch.zeros(n, device=device, dtype=torch.bool)
                used[list(used_atoms)] = True
                safe_mask = safe_mask & (~used)
            if not torch.any(safe_mask):
                return pred

            target_prob = prob[:, miss_k]
            cost = (-target_prob).masked_fill(~safe_mask, float("inf"))
            i = int(torch.argmin(cost).item())
            old_k = int(pred_k[i].item())
            if old_k == miss_k:
                continue
            pred[i] = sfac[miss_k]
            pred_k[i] = miss_k
            used_atoms.add(i)
            counts[old_k] -= 1
            counts[miss_k] += 1
            changed_any = True
        if not changed_any:
            break
        missing = (counts == 0).nonzero(as_tuple=False).view(-1).tolist()
    return pred


def _atomic_symbol_from_z(z):
    try:
        return Chem.Atom(int(z)).GetSymbol()
    except Exception:
        return ""


def _build_graph_ase(main_z_mask, pos_mask, ase_cutoff_mult=1.10, ase_extra_cutoff=0.00, ase_skin=0.00):
    from ase import Atoms
    from ase.neighborlist import NeighborList, natural_cutoffs

    z_list = [int(x) for x in main_z_mask.detach().cpu().tolist()]
    atom_symbols = [_atomic_symbol_from_z(z) for z in z_list]
    n = len(z_list)
    if n == 0:
        return atom_symbols, []

    pos = pos_mask.detach().cpu().numpy()
    atoms = Atoms(symbols=atom_symbols, positions=pos)
    cutoffs = natural_cutoffs(atoms, mult=ase_cutoff_mult)
    cutoffs = [float(c) + float(ase_extra_cutoff) for c in cutoffs]
    nl = NeighborList(cutoffs=cutoffs, self_interaction=False, bothways=True, skin=ase_skin)
    nl.update(atoms)
    degree = []
    for i in range(n):
        neigh_idx, _ = nl.get_neighbors(i)
        degree.append(int(len(neigh_idx)))
    return atom_symbols, degree


def _build_graph_rdkit(main_z_mask, pos_mask, rdkit_cov_factor=1.30):
    from rdkit import Chem as _Chem
    from rdkit.Geometry import Point3D
    from rdkit.Chem import rdDetermineBonds

    z_list = [int(x) for x in main_z_mask.detach().cpu().tolist()]
    atom_symbols = [_atomic_symbol_from_z(z) for z in z_list]
    n = len(z_list)
    if n == 0:
        return atom_symbols, []

    pos = pos_mask.detach().cpu().numpy()
    rw = _Chem.RWMol()
    for z in z_list:
        rw.AddAtom(_Chem.Atom(int(z)))
    mol = rw.GetMol()
    conf = _Chem.Conformer(n)
    for i in range(n):
        x, y, z = pos[i].tolist()
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    try:
        rdDetermineBonds.DetermineConnectivity(mol, covFactor=float(rdkit_cov_factor))
    except TypeError:
        rdDetermineBonds.DetermineConnectivity(mol)
    degree = [int(mol.GetAtomWithIdx(i).GetDegree()) for i in range(n)]
    return atom_symbols, degree


def _is_reasonable_hydrogen(atom_symbol, degree, h):
    if h < 0:
        return False
    if atom_symbol == "O":
        if h == 0:
            return True
        if h == 1:
            return degree <= 2
        if h == 2:
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


def _adjust_hydro_prediction(prob, predicted, atom_symbols, degree_list):
    sorted_indices = torch.argsort(prob, dim=1, descending=True)
    adjusted = predicted.clone()
    atom_n = min(len(atom_symbols), len(degree_list), adjusted.shape[0])
    changed = 0
    for i in range(atom_n):
        atom_symbol = atom_symbols[i]
        degree = int(degree_list[i])
        chosen = int(adjusted[i].item())
        for cand in sorted_indices[i]:
            h = int(cand.item())
            if _is_reasonable_hydrogen(atom_symbol, degree, h):
                chosen = h
                break
        if chosen != int(adjusted[i].item()):
            changed += 1
        adjusted[i] = chosen
    return adjusted, changed


def _to_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


@torch.no_grad()
def apply_one_correction_by_min_margin(prob, pred, class_labels=None):
    """
    One-shot correction: find the atom with minimal (top1 - top2) probability margin,
    and replace its prediction with top2 class.
    """
    if prob.ndim != 2 or pred.ndim != 1:
        return pred, False
    n, k = prob.shape
    if n <= 0 or k < 2:
        return pred, False

    top2_prob, top2_idx = torch.topk(prob, k=2, dim=1)
    margins = top2_prob[:, 0] - top2_prob[:, 1]
    atom_idx = int(torch.argmin(margins).item())
    second_idx = int(top2_idx[atom_idx, 1].item())

    corrected = pred.clone()
    if class_labels is None:
        corrected[atom_idx] = second_idx
    else:
        corrected[atom_idx] = class_labels[second_idx]
    return corrected, True


@torch.no_grad()
def apply_k_corrections_until_hit(prob, pred, gt, k, class_labels=None, ensure_coverage=False):
    """
    Let k define a top-k uncertain atom pool (smallest top1-top2 margins).
    Try flipping any combination of 1..k atoms in that pool to their top2 classes.
    Return the first corrected prediction that makes the whole structure correct.
    If none hits, return original pred.
    """
    if prob.ndim != 2 or pred.ndim != 1 or gt.ndim != 1:
        return pred, False, 0
    n, cls = prob.shape
    if n <= 0 or cls < 2 or k <= 0:
        return pred, False, 0

    top2_prob, top2_idx = torch.topk(prob, k=2, dim=1)
    margins = top2_prob[:, 0] - top2_prob[:, 1]
    order = torch.argsort(margins, descending=False)

    base = pred.clone()
    pool_k = min(int(k), int(n))
    candidate_atoms = [int(x.item()) for x in order[:pool_k]]
    tried = 0

    def _apply_flips(atom_indices):
        cand = base.clone()
        for atom_idx in atom_indices:
            second_idx = int(top2_idx[atom_idx, 1].item())
            if class_labels is None:
                cand[atom_idx] = int(second_idx)
            else:
                cand[atom_idx] = class_labels[second_idx]
        return cand

    def _coverage_ok(cand):
        if not (ensure_coverage and class_labels is not None):
            return True
        for lbl in class_labels:
            if not bool((cand == lbl).any().item()):
                return False
        return True

    # Try combinations with correction size from 1 to k.
    for r in range(1, pool_k + 1):
        for atom_combo in itertools.combinations(candidate_atoms, r):
            cand = _apply_flips(atom_combo)
            if not _coverage_ok(cand):
                continue
            tried += 1
            if bool((cand == gt).all().item()):
                return cand, True, tried

    return base, False, tried


@torch.no_grad()
def infer_joint(
    heavy_model,
    hydro_model,
    loader,
    device,
    restrict_to_sfac=False,
    allow_one_mismatch=False,
    allow_one_correction=False,
    allow_k_corrections=0,
    hydro_allow_k_corrections=0,
    hydro_graph_adjust=True,
    graph_builder="ase",
    ase_cutoff_mult=1.10,
    ase_extra_cutoff=0.00,
    ase_skin=0.00,
    rdkit_cov_factor=1.30,
    cif_root="",
):
    heavy_model.eval()
    hydro_model.eval()

    stats = {
        "total": 0,
        "heavy_init_correct": 0,
        "heavy_init_atom_correct": 0,
        "heavy_init_atom_total": 0,
        "heavy_correct": 0,
        "heavy_atom_correct": 0,
        "heavy_atom_total": 0,
        "hydro_eligible": 0,
        "hydro_correct": 0,
        "hydro_atom_correct": 0,
        "hydro_atom_total": 0,
        "both_correct": 0,
        "heavy_fail_forward": 0,
        "hydro_fail_forward": 0,
        "hydro_adjust_changed_atoms": 0,
        "heavy_one_correction_applied": 0,
        "heavy_k_correction_tried_structures": 0,
        "heavy_k_correction_total_trials": 0,
        "heavy_k_correction_hit": 0,
        "hydro_k_correction_tried_structures": 0,
        "hydro_k_correction_total_trials": 0,
        "hydro_k_correction_hit": 0,
        "heavy_mol_total_by_gt_len": {},
        "heavy_mol_correct_by_gt_len": {},
        "heavy_elem_pr": {},
        "hydro_group_pr": {},
    }
    heavy_init_cif_paths = []
    heavy_cif_paths = []
    both_cif_paths = []

    for data in tqdm(loader, desc="Joint infer"):
        data = data.to(device)
        stats["total"] += 1

        # heavy init accuracy by directly using z as prediction
        pred_init = data.heavy_raw_z[data.heavy_mask]
        correct_atom_init = int((pred_init == data.heavy_y).sum().item())
        mol_atom = int(data.heavy_y.shape[0])
        stats["heavy_init_atom_correct"] += correct_atom_init
        stats["heavy_init_atom_total"] += mol_atom
        stats["heavy_mol_total_by_gt_len"][mol_atom] = (
            stats["heavy_mol_total_by_gt_len"].get(mol_atom, 0) + 1
        )
        heavy_init_ok = (correct_atom_init == mol_atom)
        if heavy_init_ok:
            stats["heavy_init_correct"] += 1
            stem = _stem_from_pt_path(
                data.fname[0] if isinstance(data.fname, (list, tuple)) else data.fname
            )
            cif_path = os.path.join(cif_root, f"{stem}.cif") if cif_root else f"{stem}.cif"
            heavy_init_cif_paths.append(cif_path)

        # heavy inference (two-pass, same logic as infer_heavy_temporal.py)
        heavy_ok = False
        try:
            outputs, _ = heavy_model(data.heavy_init_z, data.pos, data.batch)
            if restrict_to_sfac:
                sfac = torch.unique(data.heavy_y)
                logits_ = outputs[:, sfac]
                prob = F.softmax(logits_, dim=-1)
                pred_idx = torch.argmax(prob, dim=1)
                predicted = sfac[pred_idx]
            else:
                prob = F.softmax(outputs, dim=-1)
                predicted = torch.argmax(prob, dim=1)

            outputs2, _ = heavy_model(predicted, data.pos, data.batch)
            outputs2 = outputs2[data.heavy_mask]
            if restrict_to_sfac:
                sfac = torch.unique(data.heavy_y)
                logits2_ = outputs2[:, sfac]
                prob2 = F.softmax(logits2_, dim=-1)
                pred_idx2 = torch.argmax(prob2, dim=1)
                pred2 = sfac[pred_idx2]
                pred2 = enforce_coverage_by_prob(prob2, sfac, pred2)
            else:
                prob2 = F.softmax(outputs2, dim=-1)
                pred2 = torch.argmax(prob2, dim=1)

            correct_atom_before = int((pred2 == data.heavy_y).sum().item())
            correction_budget = int(allow_k_corrections)
            if correction_budget <= 0 and allow_one_correction:
                correction_budget = 1
            if correction_budget > 0 and correct_atom_before < mol_atom:
                pred2, hit, used_trials = apply_k_corrections_until_hit(
                    prob2,
                    pred2,
                    data.heavy_y,
                    k=correction_budget,
                    class_labels=(sfac if restrict_to_sfac else None),
                    ensure_coverage=bool(restrict_to_sfac),
                )
                stats["heavy_k_correction_tried_structures"] += 1
                stats["heavy_k_correction_total_trials"] += int(used_trials)
                if hit:
                    stats["heavy_k_correction_hit"] += 1
                    stats["heavy_one_correction_applied"] += 1

            correct_atom = int((pred2 == data.heavy_y).sum().item())
            stats["heavy_atom_correct"] += int(correct_atom)
            stats["heavy_atom_total"] += int(mol_atom)

            all_elem = torch.unique(torch.cat([data.heavy_y, pred2], dim=0))
            for elem_z_t in all_elem:
                elem_z = int(elem_z_t.item())
                if elem_z not in stats["heavy_elem_pr"]:
                    stats["heavy_elem_pr"][elem_z] = {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0}
                gt_is_elem = data.heavy_y == elem_z
                pred_is_elem = pred2 == elem_z
                tp = int((gt_is_elem & pred_is_elem).sum().item())
                fp = int((~gt_is_elem & pred_is_elem).sum().item())
                fn = int((gt_is_elem & ~pred_is_elem).sum().item())
                gt_count = int(gt_is_elem.sum().item())
                pred_count = int(pred_is_elem.sum().item())
                stats["heavy_elem_pr"][elem_z]["tp"] += tp
                stats["heavy_elem_pr"][elem_z]["fp"] += fp
                stats["heavy_elem_pr"][elem_z]["fn"] += fn
                stats["heavy_elem_pr"][elem_z]["gt_count"] += gt_count
                stats["heavy_elem_pr"][elem_z]["pred_count"] += pred_count

            heavy_ok = (correct_atom == mol_atom) or (
                allow_one_mismatch and correct_atom == mol_atom - 1
            )
        except Exception as e:
            print("[heavy infer error]", e)
            stats["heavy_fail_forward"] += 1

        if heavy_ok:
            stats["heavy_correct"] += 1
            stats["heavy_mol_correct_by_gt_len"][mol_atom] = (
                stats["heavy_mol_correct_by_gt_len"].get(mol_atom, 0) + 1
            )
            stem = _stem_from_pt_path(
                data.fname[0] if isinstance(data.fname, (list, tuple)) else data.fname
            )
            cif_path = os.path.join(cif_root, f"{stem}.cif") if cif_root else f"{stem}.cif"
            heavy_cif_paths.append(cif_path)

        # hydro inference (same eval gate as infer_hydro_temporal.py)
        hydro_ok = False
        try:
            main_correct = (data.main_gt_hydro == data.main_z[data.hydro_mask]).all()
            if main_correct:
                stats["hydro_eligible"] += 1
                outputs_h, _ = hydro_model(data.hydro_input_z, data.pos, data.batch)
                prob_h = F.softmax(outputs_h[data.hydro_mask], dim=-1)
                _, pred_h = torch.max(prob_h, dim=1)

                if hydro_graph_adjust:
                    if graph_builder == "rdkit":
                        atom_symbols, degree_list = _build_graph_rdkit(
                            data.main_z[data.hydro_mask],
                            data.pos[data.hydro_mask],
                            rdkit_cov_factor=rdkit_cov_factor,
                        )
                    else:
                        atom_symbols, degree_list = _build_graph_ase(
                            data.main_z[data.hydro_mask],
                            data.pos[data.hydro_mask],
                            ase_cutoff_mult=ase_cutoff_mult,
                            ase_extra_cutoff=ase_extra_cutoff,
                            ase_skin=ase_skin,
                        )
                    pred_h, changed = _adjust_hydro_prediction(prob_h, pred_h, atom_symbols, degree_list)
                    stats["hydro_adjust_changed_atoms"] += int(changed)

                hydro_atom_total = int(data.hydro_y.shape[0])
                hydro_correct_atom_before = int((pred_h == data.hydro_y).sum().item())
                if hydro_allow_k_corrections > 0 and hydro_correct_atom_before < hydro_atom_total:
                    pred_h, hydro_hit, hydro_used_trials = apply_k_corrections_until_hit(
                        prob_h,
                        pred_h,
                        data.hydro_y,
                        k=int(hydro_allow_k_corrections),
                        class_labels=None,
                        ensure_coverage=False,
                    )
                    stats["hydro_k_correction_tried_structures"] += 1
                    stats["hydro_k_correction_total_trials"] += int(hydro_used_trials)
                    if hydro_hit:
                        stats["hydro_k_correction_hit"] += 1

                hydro_atom_total = int(data.hydro_y.shape[0])
                hydro_correct_atom = int((pred_h == data.hydro_y).sum().item())
                stats["hydro_atom_correct"] += hydro_correct_atom
                stats["hydro_atom_total"] += hydro_atom_total

                main_z_hydro = data.main_z[data.hydro_mask]
                pos_hydro = data.pos[data.hydro_mask]
                atom_n = min(
                    int(main_z_hydro.shape[0]),
                    int(data.hydro_y.shape[0]),
                    int(pred_h.shape[0]),
                )
                try:
                    _, rdkit_degree = _build_graph_rdkit(
                        main_z_hydro,
                        pos_hydro,
                        rdkit_cov_factor=rdkit_cov_factor,
                    )
                except Exception:
                    rdkit_degree = [0] * int(main_z_hydro.shape[0])
                for i in range(atom_n):
                    z_i = int(main_z_hydro[i].item())
                    d_i = int(rdkit_degree[i]) if i < len(rdkit_degree) else 0
                    gt_h_i = int(data.hydro_y[i].item())
                    pred_h_i = int(pred_h[i].item())
                    gt_key = _hydro_group_key(z_i, d_i, gt_h_i)
                    pred_key = _hydro_group_key(z_i, d_i, pred_h_i)

                    if gt_key not in stats["hydro_group_pr"]:
                        stats["hydro_group_pr"][gt_key] = {
                            "tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0
                        }
                    if pred_key not in stats["hydro_group_pr"]:
                        stats["hydro_group_pr"][pred_key] = {
                            "tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0
                        }

                    stats["hydro_group_pr"][gt_key]["gt_count"] += 1
                    stats["hydro_group_pr"][pred_key]["pred_count"] += 1
                    if gt_key == pred_key:
                        stats["hydro_group_pr"][gt_key]["tp"] += 1
                    else:
                        stats["hydro_group_pr"][gt_key]["fn"] += 1
                        stats["hydro_group_pr"][pred_key]["fp"] += 1

                hydro_ok = bool((pred_h == data.hydro_y).all().item())
        except Exception as e:
            print("[hydro infer error]", e)
            stats["hydro_fail_forward"] += 1

        if hydro_ok:
            stats["hydro_correct"] += 1

        if heavy_ok and hydro_ok:
            stats["both_correct"] += 1
            stem = _stem_from_pt_path(data.fname[0] if isinstance(data.fname, (list, tuple)) else data.fname)
            cif_path = os.path.join(cif_root, f"{stem}.cif") if cif_root else f"{stem}.cif"
            both_cif_paths.append(cif_path)

    return stats, both_cif_paths, heavy_init_cif_paths, heavy_cif_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", type=str, required=True)
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--heavy_model_path", type=str, required=True)
    parser.add_argument("--hydro_model_path", type=str, required=True)

    parser.add_argument("--test_years", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--pt_prefix", type=str, default="equiv_")
    parser.add_argument("--pt_suffix", type=str, default=".pt")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--seed", type=int, default=183)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--heavy_num_classes", type=int, default=0)
    parser.add_argument("--hydro_num_classes", type=int, default=0)
    parser.add_argument("--restrict_to_sfac", action="store_true")
    parser.add_argument("--allow_one_mismatch", action="store_true")
    parser.add_argument("--allow_one_correction", action="store_true")
    parser.add_argument(
        "--allow_k_corrections",
        type=int,
        default=0,
        help=(
            "Use top-k uncertain heavy atoms as correction pool. Try all top2-flip combinations "
            "with size 1..k. If any trial is fully correct, count as correct."
        ),
    )
    parser.add_argument(
        "--hydro_allow_k_corrections",
        type=int,
        default=0,
        help=(
            "Use top-k uncertain hydro atoms as correction pool. Try all top2-flip combinations "
            "with size 1..k. If any trial is fully correct, count as correct."
        ),
    )

    parser.add_argument("--disable_hydro_graph_adjust", action="store_true")
    parser.add_argument("--graph_builder", type=str, default="rdkit", choices=["ase", "rdkit"])
    parser.add_argument("--ase_cutoff_mult", type=float, default=1.10)
    parser.add_argument("--ase_extra_cutoff", type=float, default=0.00)
    parser.add_argument("--ase_skin", type=float, default=0.00)
    parser.add_argument("--rdkit_cov_factor", type=float, default=1.30)

    parser.add_argument("--heavy_noise_max", type=float, default=0.1)
    parser.add_argument("--no_dist_check", action="store_true")
    parser.add_argument("--dist_min", type=float, default=0.1)

    parser.add_argument(
        "--cif_root",
        type=str,
        default="all_cif",
    )
    parser.add_argument(
        "--exclude_disorder_txt",
        type=str,
        default="",
        help="Path to txt containing disorder CIF paths/stems to exclude from inference",
    )
    parser.add_argument(
        "--heavy_init_cif_out",
        type=str,
        default=f"heavy_init_correct_cif_{get_run_timestamp()}.txt",
    )
    parser.add_argument(
        "--heavy_cif_out",
        type=str,
        default=f"heavy_correct_cif_{get_run_timestamp()}.txt",
    )
    parser.add_argument("--both_cif_out", type=str, default=f"both_correct_cif_{get_run_timestamp()}.txt")
    parser.add_argument(
        "--heavy_elem_f1_out",
        type=str,
        default=f"heavy_elem_f1_{get_run_timestamp()}.txt",
    )
    parser.add_argument(
        "--hydro_group_f1_out",
        type=str,
        default=f"hydro_group_f1_{get_run_timestamp()}.txt",
        help="Output txt for hydro F1 by (heavy element + hydrogen count), e.g. CH2, CH3.",
    )
    parser.add_argument(
        "--hydro_label_f1_out",
        dest="hydro_group_f1_out",
        type=str,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("batch_size must be 1 for joint per-structure evaluation.")

    set_seed(args.seed, deterministic_cudnn=False)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Device:", device)

    _, test_files, missing = split_by_year_txt(
        txt_path=args.txt_path,
        pt_dir=args.pt_dir,
        test_years=tuple(args.test_years),
        pt_prefix=args.pt_prefix,
        pt_suffix=args.pt_suffix,
        strict=args.strict,
    )
    excluded_cnt = 0
    if args.exclude_disorder_txt:
        excluded_stems = load_excluded_stems(args.exclude_disorder_txt)
        old_n = len(test_files)
        test_files = [p for p in test_files if _stem_from_pt_path(p) not in excluded_stems]
        excluded_cnt = old_n - len(test_files)
    print(
        f"Test files: {len(test_files)} | Missing mapped pt: {len(missing)} | "
        f"Excluded disorder: {excluded_cnt}"
    )

    dataset, build_stats = build_joint_dataset(
        test_files,
        heavy_noise_max=args.heavy_noise_max,
        is_check_dist=(not args.no_dist_check),
        dist_min=args.dist_min,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    heavy_state = _to_state_dict(torch.load(args.heavy_model_path, map_location="cpu"))
    hydro_state = _to_state_dict(torch.load(args.hydro_model_path, map_location="cpu"))
    heavy_num_classes = args.heavy_num_classes if args.heavy_num_classes > 0 else infer_num_classes_from_state_dict(heavy_state)
    hydro_num_classes = args.hydro_num_classes if args.hydro_num_classes > 0 else infer_num_classes_from_state_dict(hydro_state)
    print(f"heavy_num_classes={heavy_num_classes} hydro_num_classes={hydro_num_classes}")

    heavy_hidden_channels = 512
    rep_heavy = TorchMD_ET(
        hidden_channels=heavy_hidden_channels,
        attn_activation="silu",
        num_heads=8,
        distance_influence="both",
    )
    out_heavy = EquivariantScalar(heavy_hidden_channels, num_classes=heavy_num_classes)
    heavy_model = TorchMD_Net(representation_model=rep_heavy, output_model=out_heavy).to(device)
    heavy_model.load_state_dict(heavy_state)

    hydro_hidden_channels = 512
    rep_hydro = TorchMD_ET(
        hidden_channels=hydro_hidden_channels,
        attn_activation="silu",
        num_heads=8,
        distance_influence="both",
    )
    out_hydro = EquivariantScalar(hydro_hidden_channels, num_classes=hydro_num_classes)
    hydro_model = TorchMD_Net(representation_model=rep_hydro, output_model=out_hydro).to(device)
    hydro_model.load_state_dict(hydro_state)

    stats, both_cif_paths, heavy_init_cif_paths, heavy_cif_paths = infer_joint(
        heavy_model=heavy_model,
        hydro_model=hydro_model,
        loader=loader,
        device=device,
        restrict_to_sfac=args.restrict_to_sfac,
        allow_one_mismatch=args.allow_one_mismatch,
        allow_one_correction=args.allow_one_correction,
        allow_k_corrections=args.allow_k_corrections,
        hydro_allow_k_corrections=args.hydro_allow_k_corrections,
        hydro_graph_adjust=(not args.disable_hydro_graph_adjust),
        graph_builder=args.graph_builder,
        ase_cutoff_mult=args.ase_cutoff_mult,
        ase_extra_cutoff=args.ase_extra_cutoff,
        ase_skin=args.ase_skin,
        rdkit_cov_factor=args.rdkit_cov_factor,
        cif_root=args.cif_root,
    )

    with open(args.heavy_init_cif_out, "w", encoding="utf-8") as f:
        for p in heavy_init_cif_paths:
            f.write(f"{p}\n")

    with open(args.heavy_cif_out, "w", encoding="utf-8") as f:
        for p in heavy_cif_paths:
            f.write(f"{p}\n")

    with open(args.both_cif_out, "w", encoding="utf-8") as f:
        for p in both_cif_paths:
            f.write(f"{p}\n")

    total = max(stats["total"], 1)
    hydro_den = max(stats["hydro_eligible"], 1)
    heavy_init_atom_den = max(stats["heavy_init_atom_total"], 1)
    heavy_atom_den = max(stats["heavy_atom_total"], 1)
    hydro_atom_den = max(stats["hydro_atom_total"], 1)
    print(
        f"[Joint] total={stats['total']} "
        f"heavy_init_correct={stats['heavy_init_correct']} ({stats['heavy_init_correct']/total*100:.2f}%) "
        f"heavy_init_atom_acc={stats['heavy_init_atom_correct']/heavy_init_atom_den*100:.2f}% "
        f"heavy_correct={stats['heavy_correct']} ({stats['heavy_correct']/total*100:.2f}%) "
        f"heavy_atom_acc={stats['heavy_atom_correct']/heavy_atom_den*100:.2f}% "
        f"hydro_correct={stats['hydro_correct']} ({stats['hydro_correct']/hydro_den*100:.2f}% on eligible={stats['hydro_eligible']}) "
        f"hydro_atom_acc={stats['hydro_atom_correct']/hydro_atom_den*100:.2f}% "
        f"both_correct={stats['both_correct']} ({stats['both_correct']/total*100:.2f}%)"
    )
    print(
        f"[Joint] heavy_fail_forward={stats['heavy_fail_forward']} "
        f"hydro_fail_forward={stats['hydro_fail_forward']} "
        f"hydro_adjust_changed_atoms={stats['hydro_adjust_changed_atoms']} "
        f"heavy_one_correction_applied={stats['heavy_one_correction_applied']} "
        f"heavy_k_correction_tried_structures={stats['heavy_k_correction_tried_structures']} "
        f"heavy_k_correction_total_trials={stats['heavy_k_correction_total_trials']} "
        f"heavy_k_correction_hit={stats['heavy_k_correction_hit']} "
        f"hydro_k_correction_tried_structures={stats['hydro_k_correction_tried_structures']} "
        f"hydro_k_correction_total_trials={stats['hydro_k_correction_total_trials']} "
        f"hydro_k_correction_hit={stats['hydro_k_correction_hit']}"
    )
    elem_lines = [
        "atomic_num\telement\tf1(%)\tprecision(%)\trecall(%)\tgt_count\tpred_count\ttp\tfp\tfn\tsupport"
    ]
    for elem_z in sorted(stats["heavy_elem_pr"].keys()):
        elem_stat = stats["heavy_elem_pr"][elem_z]
        tp = int(elem_stat["tp"])
        fp = int(elem_stat["fp"])
        fn = int(elem_stat["fn"])
        gt_count = int(elem_stat["gt_count"])
        pred_count = int(elem_stat["pred_count"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2.0 * tp) / max((2 * tp + fp + fn), 1)
        support = gt_count
        elem_name = _elem_symbol(elem_z)
        print(
            f"[ElemF1] {elem_name}(Z={elem_z}) "
            f"f1={f1*100:.2f}% precision={precision*100:.2f}% recall={recall*100:.2f}% "
            f"(gt={gt_count} pred={pred_count} tp={tp} fp={fp} fn={fn})"
        )
        elem_lines.append(
            f"{elem_z}\t{elem_name}\t{f1*100:.4f}\t{precision*100:.4f}\t{recall*100:.4f}\t"
            f"{gt_count}\t{pred_count}\t{tp}\t{fp}\t{fn}\t{support}"
        )

    with open(args.heavy_elem_f1_out, "w", encoding="utf-8") as f:
        f.write("\n".join(elem_lines) + "\n")

    hydro_lines = [
        "atomic_num\telement\tdegree\thydro_count\tgroup\tf1(%)\tprecision(%)\trecall(%)\tgt_count\tpred_count\ttp\tfp\tfn\tsupport"
    ]
    for z_i, d_i, h_i in sorted(stats["hydro_group_pr"].keys(), key=lambda x: (int(x[0]), int(x[1]), int(x[2]))):
        h_stat = stats["hydro_group_pr"][(z_i, d_i, h_i)]
        tp = int(h_stat["tp"])
        fp = int(h_stat["fp"])
        fn = int(h_stat["fn"])
        gt_count = int(h_stat["gt_count"])
        pred_count = int(h_stat["pred_count"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2.0 * tp) / max((2 * tp + fp + fn), 1)
        support = gt_count
        elem = "ALL" if (int(z_i) == 0 and int(h_i) == 0) else _elem_symbol(z_i)
        group = _hydro_group_name(z_i, d_i, h_i)
        print(
            f"[HydroGroupF1] {group}(Z={z_i},deg={d_i}) "
            f"f1={f1*100:.2f}% precision={precision*100:.2f}% recall={recall*100:.2f}% "
            f"(gt={gt_count} pred={pred_count} tp={tp} fp={fp} fn={fn})"
        )
        hydro_lines.append(
            f"{z_i}\t{elem}\t{d_i}\t{h_i}\t{group}\t{f1*100:.4f}\t{precision*100:.4f}\t{recall*100:.4f}\t"
            f"{gt_count}\t{pred_count}\t{tp}\t{fp}\t{fn}\t{support}"
        )

    with open(args.hydro_group_f1_out, "w", encoding="utf-8") as f:
        f.write("\n".join(hydro_lines) + "\n")

    print(f"[BuildStats] {build_stats}")
    print(f"[HeavyInitCorrect] saved {len(heavy_init_cif_paths)} paths -> {args.heavy_init_cif_out}")
    print(f"[HeavyCorrect] saved {len(heavy_cif_paths)} paths -> {args.heavy_cif_out}")
    print(f"[BothCorrect] saved {len(both_cif_paths)} paths -> {args.both_cif_out}")
    print(f"[HeavyElemF1] saved {len(elem_lines)-1} elements -> {args.heavy_elem_f1_out}")
    print(f"[HydroGroupF1] saved {len(hydro_lines)-1} groups -> {args.hydro_group_f1_out}")


if __name__ == "__main__":
    main()
