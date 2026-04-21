"""Shared heavy-atom inference helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from crystalx_infer.common.chem import atomic_num_from_symbol


def parse_sfac_unit_from_shelx(file_path):
    sfac = None
    unit = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file_obj:
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
        raise ValueError(f"Missing SFAC/UNIT in {file_path}")
    if len(sfac) != len(unit):
        raise ValueError(f"SFAC/UNIT length mismatch in {file_path}: {len(sfac)} vs {len(unit)}")

    return sfac, unit


def parse_sfac_unit_from_ins(ins_path):
    return parse_sfac_unit_from_shelx(ins_path)


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
            atomic_num = atomic_num_from_symbol(symbol)
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


def predict_class_labels(logits, class_labels=None):
    if class_labels is None:
        prob = F.softmax(logits, dim=-1)
        predicted = torch.argmax(prob, dim=1)
    else:
        prob = F.softmax(logits[:, class_labels], dim=-1)
        predicted = class_labels[torch.argmax(prob, dim=1)]
    return prob, predicted


@torch.no_grad()
def run_two_pass_heavy_prediction(
    model,
    input_z,
    pos,
    batch,
    mask,
    candidate_elements=None,
    enforce_element_coverage=False,
):
    logits = model(input_z, pos, batch)
    _, predicted = predict_class_labels(logits, candidate_elements)

    logits_second = model(predicted, pos, batch)[mask]
    prob_second, predicted_second = predict_class_labels(logits_second, candidate_elements)

    if candidate_elements is not None and enforce_element_coverage:
        predicted_second = enforce_coverage_by_prob(
            prob_second,
            candidate_elements,
            predicted_second.clone(),
        )

    return prob_second, predicted_second, predicted
