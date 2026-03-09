import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from cctbx.xray.structure import structure
from iotbx.shelx import crystal_symmetry_from_ins
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from crystalx_infer.common.utils import (
    copy_file,
    get_equiv_pos2,
    load_shelxt,
    update_shelxt,
)
from crystalx_infer.models.noise_output_model import EquivariantScalar
from crystalx_infer.models.torchmd_et import TorchMD_ET
from crystalx_infer.models.torchmd_net import TorchMD_Net


HIDDEN_CHANNELS = 512
NUM_CLASSES = 98


def parse_sfac_unit_from_ins(ins_path):
    sfac = None
    unit = None

    with open(ins_path, "r", encoding="utf-8", errors="ignore") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue

            upper_line = line.upper()
            if upper_line.startswith("SFAC "):
                sfac = [item.capitalize() for item in line.split()[1:]]
            elif upper_line.startswith("UNIT "):
                unit = [int(float(item)) for item in line.split()[1:]]

    if not sfac or not unit:
        raise ValueError(f"Missing SFAC/UNIT in {ins_path}")
    if len(sfac) != len(unit):
        raise ValueError(f"SFAC/UNIT length mismatch in {ins_path}: {len(sfac)} vs {len(unit)}")

    return sfac, unit


def trim_shelxt_pred_for_unit_divisibility(real_frac, shelxt_pred, isotropy_list, non_h_unit_total):
    pred_n = int(len(shelxt_pred))
    unit_n = int(non_h_unit_total)
    if pred_n <= 0 or unit_n <= 0:
        return real_frac, shelxt_pred, isotropy_list, 0
    if (pred_n % unit_n == 0) or (unit_n % pred_n == 0):
        return real_frac, shelxt_pred, isotropy_list, 0

    new_len = pred_n
    while new_len > 0 and not ((new_len % unit_n == 0) or (unit_n % new_len == 0)):
        new_len -= 1

    trim_n = pred_n - new_len
    if trim_n <= 0:
        return real_frac, shelxt_pred, isotropy_list, 0

    real_frac = real_frac[:new_len]
    shelxt_pred = shelxt_pred[:new_len]
    if isotropy_list is not None and len(isotropy_list) >= new_len:
        isotropy_list = isotropy_list[:new_len]

    print(
        f"[WARN] SHELXT pred divisibility adjust: pred_n={pred_n}, nonH_unit_total={unit_n}, "
        f"trim_tail={trim_n} -> pred_n={new_len}"
    )
    return real_frac, shelxt_pred, isotropy_list, trim_n


def build_sorted_init_z_from_ratio(sfac, unit, target_len):
    if target_len <= 0:
        return []

    valid_entries = []
    for symbol, count in zip(sfac, unit):
        if symbol.upper() == "H":
            continue
        try:
            atomic_num = Chem.Atom(symbol).GetAtomicNum()
        except Exception:
            continue
        if count > 0:
            valid_entries.append((atomic_num, int(count)))

    if not valid_entries:
        raise ValueError("No valid positive non-H SFAC/UNIT entries")

    counts = np.array([count for _, count in valid_entries], dtype=np.float64)
    weights = counts / np.sum(counts)
    raw_alloc = weights * float(target_len)
    alloc = np.floor(raw_alloc).astype(np.int64)

    remain = int(target_len - int(np.sum(alloc)))
    if remain > 0:
        frac = raw_alloc - alloc
        for idx in np.argsort(-frac)[:remain]:
            alloc[idx] += 1

    init_z = []
    for (atomic_num, _), count in zip(valid_entries, alloc.tolist()):
        if count > 0:
            init_z.extend([int(atomic_num)] * int(count))

    if len(init_z) < target_len:
        min_z = min(atomic_num for atomic_num, _ in valid_entries)
        init_z.extend([int(min_z)] * (target_len - len(init_z)))
    elif len(init_z) > target_len:
        init_z = init_z[:target_len]

    return sorted(init_z, reverse=True)


def deduplicate_cartesian_positions(expanded_cart, expanded_z):
    real_cart = []
    z = []

    for idx in range(expanded_cart.shape[0]):
        position = expanded_cart[idx].tolist()
        if position not in real_cart:
            real_cart.append(position)
            z.append(expanded_z[idx])

    return real_cart, z


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


@torch.no_grad()
def enforce_coverage_by_prob(prob, sfac, pred):
    num_atoms, num_elements = prob.shape
    if num_atoms == 0 or num_elements == 0 or num_atoms < num_elements:
        return pred

    elem2idx = {int(sfac[idx].item()): idx for idx in range(num_elements)}
    pred_idx = torch.tensor(
        [elem2idx[int(value.item())] for value in pred],
        device=pred.device,
        dtype=torch.long,
    )
    counts = torch.bincount(pred_idx, minlength=num_elements).clone()
    used_atoms = set()
    missing = (counts == 0).nonzero(as_tuple=False).view(-1).tolist()

    while missing:
        changed = False
        for miss_idx in missing:
            safe_mask = counts[pred_idx] > 1
            if used_atoms:
                used_mask = torch.zeros(num_atoms, device=pred.device, dtype=torch.bool)
                used_mask[list(used_atoms)] = True
                safe_mask = safe_mask & (~used_mask)
            if not torch.any(safe_mask):
                return pred

            target_prob = prob[:, miss_idx]
            cost = (-target_prob).masked_fill(~safe_mask, float("inf"))
            atom_idx = int(torch.argmin(cost).item())

            old_idx = int(pred_idx[atom_idx].item())
            if old_idx == miss_idx:
                continue

            pred[atom_idx] = sfac[miss_idx]
            pred_idx[atom_idx] = miss_idx
            used_atoms.add(atom_idx)
            counts[old_idx] -= 1
            counts[miss_idx] += 1
            changed = True

        if not changed:
            break
        missing = (counts == 0).nonzero(as_tuple=False).view(-1).tolist()

    return pred


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    origin_ins_path = f"{args.work_dir}/{args.fname}.ins"
    res_file_path = f"{args.work_dir}/{args.fname}_a.res"
    hkl_file_path = f"{args.work_dir}/{args.fname}_a.hkl"
    new_res_file_path = f"{args.work_dir}/{args.fname}_AI.ins"
    new_hkl_file_path = f"{args.work_dir}/{args.fname}_AI.hkl"

    real_frac, shelxt_pred, _, isotropy_list = load_shelxt(res_file_path, is_check_sfac=True)

    sfac0 = None
    unit0 = None
    non_h_unit_total = 0
    try:
        sfac0, unit0 = parse_sfac_unit_from_ins(origin_ins_path)
        non_h_unit_total = sum(
            int(count) for symbol, count in zip(sfac0, unit0) if symbol.upper() != "H"
        )
    except Exception as exc:
        print(f"[WARN] Cannot parse SFAC/UNIT for divisibility check, skip trim: {exc}")

    if non_h_unit_total > 0:
        real_frac, shelxt_pred, isotropy_list, _ = trim_shelxt_pred_for_unit_divisibility(
            real_frac,
            shelxt_pred,
            isotropy_list,
            non_h_unit_total,
        )

    real_num = len(shelxt_pred)
    crystal_symmetry = crystal_symmetry_from_ins.extract_from(res_file_path)
    coarse_structure = structure(crystal_symmetry=crystal_symmetry)

    real_frac = np.array(real_frac)
    expanded_cart, expanded_symbols = get_equiv_pos2(
        real_frac,
        shelxt_pred,
        coarse_structure,
        radius=3.2,
    )
    expanded_z = [Chem.Atom(symbol.capitalize()).GetAtomicNum() for symbol in expanded_symbols]

    real_cart, z = deduplicate_cartesian_positions(expanded_cart, expanded_z)
    if len(z) >= real_num:
        if sfac0 is not None and unit0 is not None:
            z = build_sorted_init_z_from_ratio(sfac0, unit0, real_num) + z[real_num:]
        else:
            print("[WARN] SFAC/UNIT init fallback to SHELXT z: SFAC/UNIT unavailable.")

    mask = torch.zeros(len(z), dtype=torch.bool)
    mask[: len(shelxt_pred)] = True

    test_loader = DataLoader(
        [Data(z=torch.tensor(z), pos=torch.from_numpy(np.array(real_cart, dtype=np.float32)), mask=mask)],
        batch_size=1,
        shuffle=False,
    )
    model = build_model(args.model_path, device)

    predicted_symbols = []
    with torch.inference_mode():
        for data in tqdm(test_loader):
            data = data.to(device)

            logits = model(data.z, data.pos, data.batch)
            sfac = torch.unique(data.z)
            logits = F.softmax(logits[:, sfac], dim=-1)
            predicted = sfac[torch.argmax(logits, dim=1)]

            logits = model(predicted, data.pos, data.batch)
            logits = F.softmax(logits[data.mask][:, sfac], dim=-1)
            predicted = sfac[torch.argmax(logits, dim=1)]
            predicted = enforce_coverage_by_prob(logits, sfac, predicted)
            predicted_symbols = [Chem.Atom(int(item)).GetSymbol() for item in predicted.cpu().tolist()]

    copy_file(hkl_file_path, new_hkl_file_path)
    update_shelxt(
        res_file_path,
        new_res_file_path,
        predicted_symbols,
        no_sfac=True,
        refine_round=16,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Non-hydro Atom Classifier")
    parser.add_argument("--fname", type=str, default="sample2", help="file name")
    parser.add_argument("--work_dir", type=str, default="work_dir", help="work directory")
    parser.add_argument(
        "--model_path",
        type=str,
        default="final_main_model_add_no_noise_fold_3.pth",
        help="model path",
    )
    main(parser.parse_args())
