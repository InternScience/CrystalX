"""Dataset builders for CrystalX inference evaluation scripts."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from scipy.spatial.distance import cdist
from tqdm import tqdm

from crystalx_infer.common.chem import atomic_num_list


def _pyg_data_class():
    from torch_geometric.data import Data

    return Data


def _optional_noise_passes_threshold(mol_info, max_abs_threshold):
    if "noise_list" not in mol_info:
        return True

    noise = np.asarray(mol_info["noise_list"])
    if noise.size == 0:
        return True
    return float(np.max(np.abs(noise))) <= float(max_abs_threshold)


def _load_dataset_pt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def split_by_year_txt(
    txt_path: str,
    pt_dir: str,
    test_years=(2022, 2023, 2024),
    pt_prefix="equiv_",
    pt_suffix=".pt",
    strict=False,
    split_mode="year",
    random_train_ratio=0.8,
    random_test_ratio=0.2,
    split_seed=150,
):
    split_mode = str(split_mode).strip().lower()
    if split_mode not in {"year", "random"}:
        raise ValueError(f"Unsupported split_mode={split_mode!r}. Expected 'year' or 'random'.")

    if split_mode == "random":
        ratio_sum = float(random_train_ratio) + float(random_test_ratio)
        if float(random_train_ratio) <= 0 or float(random_test_ratio) <= 0:
            raise ValueError("random_train_ratio and random_test_ratio must both be positive.")
        if abs(ratio_sum - 1.0) > 1e-8:
            raise ValueError(
                "random_train_ratio and random_test_ratio must sum to 1.0, "
                f"got {random_train_ratio} + {random_test_ratio} = {ratio_sum}."
            )

    test_years = set(str(year) for year in test_years)

    train_files, test_files = [], []
    listed_files = []
    missing = []
    seen = set()

    with open(txt_path, "r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"[WARN] txt line {line_no} format error: {line}")
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

            listed_files.append(pt_path)
            if split_mode == "year":
                if year in test_years:
                    test_files.append(pt_path)
                else:
                    train_files.append(pt_path)

    extra = []
    if not strict:
        listed_set = set(listed_files)
        all_pt = [
            os.path.join(pt_dir, filename)
            for filename in os.listdir(pt_dir)
            if filename.endswith(pt_suffix)
        ]
        extra = [path for path in all_pt if path not in listed_set]
        if split_mode == "year":
            train_files += extra

    if split_mode == "random":
        candidate_files = sorted(set(listed_files) | set(extra))
        if not candidate_files:
            return [], [], missing

        shuffled = sorted(candidate_files)
        random.Random(int(split_seed)).shuffle(shuffled)
        if len(shuffled) == 1:
            return [], shuffled, missing

        test_count = int(round(len(shuffled) * float(random_test_ratio)))
        test_count = max(1, min(test_count, len(shuffled) - 1))
        test_set = set(shuffled[:test_count])
        train_files = sorted(path for path in shuffled if path not in test_set)
        test_files = sorted(test_set)

    return train_files, test_files, missing


def build_heavy_eval_dataset(file_list, is_eval=True, is_check_dist=True):
    Data = _pyg_data_class()

    dataset = []
    stats = {
        "kept": 0,
        "dist_drop": 0,
        "noise_drop": 0,
        "element_drop": 0,
    }

    for fname in tqdm(file_list, desc="Build heavy dataset"):
        mol_info = _load_dataset_pt(fname)

        if not _optional_noise_passes_threshold(mol_info, 0.1):
            stats["noise_drop"] += 1
            continue

        try:
            atomic_z = atomic_num_list([item.capitalize() for item in mol_info["z"]])
            y = atomic_num_list([item.capitalize() for item in mol_info["gt"]])
        except Exception as exc:
            print("[Element error]", exc, "in", fname)
            stats["element_drop"] += 1
            continue

        raw_cart = mol_info["pos"]
        real_cart = []
        z_num = []
        for idx in range(raw_cart.shape[0]):
            position = raw_cart[idx].tolist()
            if position not in real_cart:
                real_cart.append(position)
                z_num.append(atomic_z[idx])
        real_cart = np.array(real_cart, dtype=np.float32)

        mask = np.zeros(len(z_num), dtype=np.int64)
        mask[: len(y)] = 1
        mask = torch.from_numpy(mask).bool()

        if is_check_dist:
            distance_matrix = cdist(real_cart, real_cart)
            distance_matrix = distance_matrix + 10 * np.eye(distance_matrix.shape[0])
            if np.min(distance_matrix) < 0.1:
                stats["dist_drop"] += 1
                continue

        all_z = [sorted(y, reverse=True)] if is_eval else [sorted(y, reverse=True), z_num[: len(y)]]
        y_tensor = torch.tensor(y, dtype=torch.long)
        pos_tensor = torch.from_numpy(real_cart)

        for pz in all_z:
            pz = list(pz) + z_num[len(y) :]
            dataset.append(
                Data(
                    z=torch.tensor(pz, dtype=torch.long),
                    y=y_tensor,
                    pos=pos_tensor,
                    fname=fname,
                    mask=mask,
                )
            )
        stats["kept"] += 1

    return dataset, stats


def build_hydro_eval_dataset(file_list, is_check_dist=False, is_filter=False):
    Data = _pyg_data_class()

    dataset = []
    stats = {
        "kept": 0,
        "dist_drop": 0,
        "skip_c_h_gt3": 0,
        "element_drop": 0,
    }
    max_h = -1

    for fname in tqdm(file_list, desc="Build hydro dataset"):
        mol_info = _load_dataset_pt(fname)

        try:
            z_equiv = atomic_num_list([item.capitalize() for item in mol_info["equiv_gt"]])
            gt_main = atomic_num_list([item.capitalize() for item in mol_info["gt"]])
        except Exception:
            stats["element_drop"] += 1
            continue

        hydro_num = mol_info["hydro_gt"]
        if isinstance(hydro_num, torch.Tensor):
            hydro_num = hydro_num.detach().cpu().tolist()
        elif isinstance(hydro_num, np.ndarray):
            hydro_num = hydro_num.tolist()
        hydro_num = [int(value) for value in hydro_num]

        pair_n = min(len(gt_main), len(hydro_num))
        if any(gt_main[idx] == 6 and hydro_num[idx] > 3 for idx in range(pair_n)):
            stats["skip_c_h_gt3"] += 1
            continue

        max_h = max(max_h, max(hydro_num))

        if is_filter:
            if max(hydro_num) > 4:
                continue
            if 6 not in z_equiv or 7 not in z_equiv or 8 not in z_equiv:
                continue

        raw_cart = mol_info["pos"]
        real_cart = []
        z_num = []
        main_z = []
        for idx in range(raw_cart.shape[0]):
            position = raw_cart[idx].tolist()
            if position not in real_cart:
                real_cart.append(position)
                z_num.append(z_equiv[idx])
                main_z.append(z_equiv[idx])
        real_cart = np.array(real_cart, dtype=np.float32)

        mask = np.zeros(len(z_num), dtype=np.int64)
        mask[: len(hydro_num)] = 1
        mask = torch.from_numpy(mask).bool()

        if is_check_dist:
            distance_matrix = cdist(real_cart, real_cart)
            distance_matrix = distance_matrix + 10 * np.eye(distance_matrix.shape[0])
            if np.min(distance_matrix) < 0.1:
                stats["dist_drop"] += 1
                continue

        dataset.append(
            Data(
                z=torch.tensor(z_num, dtype=torch.long),
                main_z=torch.tensor(main_z, dtype=torch.long),
                y=torch.tensor(hydro_num, dtype=torch.long),
                pos=torch.from_numpy(real_cart),
                main_gt=torch.tensor(gt_main, dtype=torch.long),
                fname=fname,
                mask=mask,
            )
        )
        stats["kept"] += 1

    return dataset, stats, max_h


def build_joint_dataset(file_list, heavy_noise_max=0.1, is_check_dist=True, dist_min=0.1):
    Data = _pyg_data_class()

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
    required = {"z", "equiv_gt", "gt", "hydro_gt", "pos"}

    for fname in tqdm(file_list, desc="Build joint dataset"):
        try:
            mol = _load_dataset_pt(fname)
        except Exception:
            stats["drop_missing_keys"] += 1
            continue

        if not required.issubset(set(mol.keys())):
            stats["drop_missing_keys"] += 1
            continue

        if not _optional_noise_passes_threshold(mol, heavy_noise_max):
            stats["drop_noise"] += 1
            continue

        try:
            z_heavy = atomic_num_list(mol["z"])
            z_equiv = atomic_num_list(mol["equiv_gt"])
            gt_main = atomic_num_list(mol["gt"])
        except Exception:
            stats["drop_element"] += 1
            continue

        hydro_gt = mol["hydro_gt"]
        if isinstance(hydro_gt, torch.Tensor):
            hydro_gt = hydro_gt.detach().cpu().tolist()
        elif isinstance(hydro_gt, np.ndarray):
            hydro_gt = hydro_gt.tolist()
        hydro_gt = [int(value) for value in hydro_gt]

        pair_n = min(len(gt_main), len(hydro_gt))
        if any(gt_main[idx] == 6 and hydro_gt[idx] > 3 for idx in range(pair_n)):
            stats["drop_c_h_gt3"] += 1
            continue

        pos_arr = mol["pos"]
        real_cart = []
        keep_idx = []
        for idx in range(pos_arr.shape[0]):
            position = pos_arr[idx].tolist()
            if position not in real_cart:
                real_cart.append(position)
                keep_idx.append(idx)
        real_cart = np.array(real_cart, dtype=np.float32)
        z_heavy_dedup = [z_heavy[idx] for idx in keep_idx]
        z_equiv_dedup = [z_equiv[idx] for idx in keep_idx]

        if is_check_dist:
            distance_matrix = cdist(real_cart, real_cart)
            distance_matrix = distance_matrix + 10 * np.eye(distance_matrix.shape[0])
            if np.min(distance_matrix) < dist_min:
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

    return dataset, stats
