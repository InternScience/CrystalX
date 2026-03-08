import os
import json
from rdkit import Chem
from iotbx.shelx import crystal_symmetry_from_ins
from cctbx.xray.structure import structure
from crystalx_infer.common.utils import *
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from crystalx_infer.models.torchmd_et import TorchMD_ET
from crystalx_infer.models.noise_output_model import EquivariantScalar
from crystalx_infer.models.torchmd_net import TorchMD_Net
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import argparse

HIDDEN_CHANNELS = 512


def parse_sfac_unit_from_ins(ins_path):
    sfac = None
    unit = None
    with open(ins_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            up = s.upper()
            if up.startswith("SFAC "):
                sfac = [x.capitalize() for x in s.split()[1:]]
            elif up.startswith("UNIT "):
                unit = [int(float(x)) for x in s.split()[1:]]
    if not sfac or not unit:
        raise ValueError(f"Missing SFAC/UNIT in {ins_path}")
    if len(sfac) != len(unit):
        raise ValueError(f"SFAC/UNIT length mismatch in {ins_path}: {len(sfac)} vs {len(unit)}")
    return sfac, unit


def trim_shelxt_pred_for_unit_divisibility(real_frac, shelxt_pred, isotropy_list, non_h_unit_total):
    """
    Enforce divisibility relation between:
      a = len(shelxt_pred)
      b = non-H total count in UNIT
    Keep trimming tail atoms until (a % b == 0) or (b % a == 0).
    """
    a = int(len(shelxt_pred))
    b = int(non_h_unit_total)
    if a <= 0 or b <= 0:
        return real_frac, shelxt_pred, isotropy_list, 0
    if (a % b == 0) or (b % a == 0):
        return real_frac, shelxt_pred, isotropy_list, 0

    new_len = a
    while new_len > 0 and not ((new_len % b == 0) or (b % new_len == 0)):
        new_len -= 1

    trim_n = a - new_len
    if trim_n <= 0:
        return real_frac, shelxt_pred, isotropy_list, 0

    real_frac = real_frac[:new_len]
    shelxt_pred = shelxt_pred[:new_len]
    if isotropy_list is not None and len(isotropy_list) >= new_len:
        isotropy_list = isotropy_list[:new_len]

    print(
        f"[WARN] SHELXT pred divisibility adjust: pred_n={a}, nonH_unit_total={b}, "
        f"trim_tail={trim_n} -> pred_n={new_len}"
    )
    return real_frac, shelxt_pred, isotropy_list, trim_n


def build_sorted_init_z_from_ratio(sfac, unit, target_len):
    if target_len <= 0:
        return []

    valid = []
    for sym, cnt in zip(sfac, unit):
        # Heavy-atom initialization: exclude hydrogen from SFAC/UNIT ratio.
        if sym.upper() == "H":
            continue
        try:
            z = Chem.Atom(sym).GetAtomicNum()
        except Exception:
            continue
        if cnt > 0:
            valid.append((z, int(cnt)))
    if not valid:
        raise ValueError("No valid positive non-H SFAC/UNIT entries")

    counts = np.array([v[1] for v in valid], dtype=np.float64)
    weights = counts / np.sum(counts)
    raw = weights * float(target_len)
    alloc = np.floor(raw).astype(np.int64)
    remain = int(target_len - int(np.sum(alloc)))
    if remain > 0:
        frac = raw - alloc
        order = np.argsort(-frac)
        for i in order[:remain]:
            alloc[i] += 1

    init_z = []
    for (z, _), c in zip(valid, alloc.tolist()):
        if c > 0:
            init_z.extend([int(z)] * int(c))

    if len(init_z) < target_len:
        # Fallback padding with smallest atomic number to keep length stable.
        min_z = min(v[0] for v in valid)
        init_z.extend([int(min_z)] * (target_len - len(init_z)))
    elif len(init_z) > target_len:
        init_z = init_z[:target_len]

    init_z = sorted(init_z, reverse=True)
    return init_z


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

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fname = args.fname
    work_dir = args.work_dir
    model_path = args.model_path

    origin_ins_path = f'{work_dir}/{fname}.ins'
    res_file_path = f'{work_dir}/{fname}_a.res'
    hkl_file_path = f'{work_dir}/{fname}_a.hkl'
    new_res_file_path = f'{work_dir}/{fname}_AI.ins'
    new_hkl_file_path = f'{work_dir}/{fname}_AI.hkl'

    real_frac, shelxt_pred, _, isotropy_list = load_shelxt(res_file_path,is_check_sfac=True)

    sfac0 = None
    unit0 = None
    non_h_unit_total = 0
    try:
        sfac0, unit0 = parse_sfac_unit_from_ins(origin_ins_path)
        non_h_unit_total = sum(
            int(cnt) for sym, cnt in zip(sfac0, unit0) if sym.upper() != "H"
        )
    except Exception as e:
        print(f"[WARN] Cannot parse SFAC/UNIT for divisibility check, skip trim: {e}")

    if non_h_unit_total > 0:
        real_frac, shelxt_pred, isotropy_list, _ = trim_shelxt_pred_for_unit_divisibility(
            real_frac, shelxt_pred, isotropy_list, non_h_unit_total
        )

    real_num = len(shelxt_pred)
    # print(shelxt_pred)

    sym = crystal_symmetry_from_ins.extract_from(res_file_path)
    coarse_structure = structure(crystal_symmetry=sym)

    real_frac = np.array(real_frac)
    _real_cart, _z = get_equiv_pos2(real_frac, shelxt_pred, coarse_structure, radius=3.2)

    _z = [item.capitalize() for item in _z]
    _z = [Chem.Atom(item).GetAtomicNum() for item in _z]

    # _z = sorted(_z[:real_num], reverse=True) + _z[real_num:]

    real_cart = []
    z = []
    for i in range(_real_cart.shape[0]):
        if _real_cart[i].tolist() not in real_cart:
            real_cart.append(_real_cart[i].tolist())
            z.append(_z[i])

    # Initialize first real_num atom types from original .ins SFAC/UNIT ratio,
    # sorted descending by atomic number (same spirit as infer joint init).
    if len(z) >= real_num:
        if sfac0 is not None and unit0 is not None:
            init_main_z = build_sorted_init_z_from_ratio(sfac0, unit0, real_num)
            z = init_main_z + z[real_num:]
        else:
            print("[WARN] SFAC/UNIT init fallback to SHELXT z: SFAC/UNIT unavailable.")

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
    num_classes = 98
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

            sfac = torch.unique(data.z)
            logits = logits[:,sfac]
            logits = F.softmax(logits, dim = -1)

            _, predicted = torch.max(logits, dim=1)
            predicted = sfac[predicted]

            outputs = model(predicted, data.pos, data.batch)
            logits = outputs[data.mask]
            logits = logits[:,sfac]
            logits = F.softmax(logits, dim = -1)
            _, predicted = torch.max(logits, dim=1)
            predicted = sfac[predicted]
            predicted = enforce_coverage_by_prob(logits, sfac, predicted)

            predicted = predicted.cpu()
            predicted = predicted.to('cpu').tolist()
            
            predicted = [Chem.Atom(item).GetSymbol() for item in predicted]

    copy_file(hkl_file_path, new_hkl_file_path)
    update_shelxt(res_file_path, new_res_file_path, predicted, no_sfac = True,refine_round=16)

    # print(predicted)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Non-hydro Atom Classifier')
    parser.add_argument('--fname', type=str, default='sample2', help='file name')
    parser.add_argument('--work_dir', type=str, default='work_dir', help='work directory')
    parser.add_argument('--model_path', type=str, default='final_main_model_add_no_noise_fold_3.pth', help='model path')
    args = parser.parse_args()
    main(args)
