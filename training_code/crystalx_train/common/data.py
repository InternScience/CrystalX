from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class SplitSpec:
    """Dataset split configuration based on txt metadata and pt file layout."""

    txt_path: str
    pt_dir: str
    test_years: tuple[int, ...] = (2022, 2023, 2024)
    pt_prefix: str = "equiv_"
    pt_suffix: str = ".pt"
    strict: bool = False
    split_mode: str = "year"
    random_train_ratio: float = 0.8
    random_test_ratio: float = 0.2
    split_seed: int = 150

    def normalized_test_years(self) -> set[str]:
        return {str(year) for year in self.test_years}

    def normalized_split_mode(self) -> str:
        return self.split_mode.strip().lower()


def _validate_random_ratios(train_ratio: float, test_ratio: float) -> None:
    total = float(train_ratio) + float(test_ratio)
    if float(train_ratio) <= 0 or float(test_ratio) <= 0:
        raise ValueError("random_train_ratio and random_test_ratio must both be positive.")
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "random_train_ratio and random_test_ratio must sum to 1.0, "
            f"got {train_ratio} + {test_ratio} = {total}."
        )


def load_dataset_pt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _random_split_files(
    candidate_files: list[str],
    *,
    train_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> tuple[list[str], list[str]]:
    _validate_random_ratios(train_ratio, test_ratio)
    if not candidate_files:
        return [], []

    shuffled = sorted(candidate_files)
    random.Random(split_seed).shuffle(shuffled)

    if len(shuffled) == 1:
        return [], shuffled

    test_count = int(round(len(shuffled) * float(test_ratio)))
    test_count = max(1, min(test_count, len(shuffled) - 1))
    test_set = set(shuffled[:test_count])
    train_files = sorted(path for path in shuffled if path not in test_set)
    test_files = sorted(test_set)
    return train_files, test_files


def split_by_year_txt(spec: SplitSpec) -> tuple[list[str], list[str], list[str]]:
    split_mode = spec.normalized_split_mode()
    if split_mode not in {"year", "random"}:
        raise ValueError(f"Unsupported split_mode={spec.split_mode!r}. Expected 'year' or 'random'.")

    test_years = spec.normalized_test_years()
    pt_dir = Path(spec.pt_dir)

    train_files: list[str] = []
    test_files: list[str] = []
    listed_files: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()

    with open(spec.txt_path, "r", encoding="utf-8") as file_obj:
        for line_num, line in enumerate(file_obj, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"[WARN] malformed line {line_num} in {spec.txt_path}: {line}")
                continue

            year = parts[0]
            cif_stem = Path(parts[-1]).stem
            pt_path = str(pt_dir / f"{spec.pt_prefix}{cif_stem}{spec.pt_suffix}")

            if pt_path in seen:
                continue
            seen.add(pt_path)

            if not Path(pt_path).exists():
                missing.append(pt_path)
                continue

            listed_files.append(pt_path)
            if split_mode == "year":
                if year in test_years:
                    test_files.append(pt_path)
                else:
                    train_files.append(pt_path)

    extra_files: list[str] = []
    if not spec.strict and pt_dir.exists():
        listed_path_set = set(listed_files)
        extra_files = sorted(
            str(path)
            for path in pt_dir.iterdir()
            if path.is_file()
            and path.name.endswith(spec.pt_suffix)
            and str(path) not in listed_path_set
        )
        if split_mode == "year":
            train_files.extend(extra_files)

    if split_mode == "random":
        candidate_files = sorted(set(listed_files) | set(extra_files))
        return _random_split_files(
            candidate_files,
            train_ratio=spec.random_train_ratio,
            test_ratio=spec.random_test_ratio,
            split_seed=spec.split_seed,
        ) + (missing,)

    return train_files, test_files, missing


def deduplicate_positions(
    atom_numbers: Sequence[int],
    positions: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    unique_atoms: list[int] = []
    unique_positions: list[list[float]] = []
    seen_positions: set[tuple[float, ...]] = set()

    for atom_number, position in zip(atom_numbers, positions):
        key = tuple(float(value) for value in np.asarray(position).tolist())
        if key in seen_positions:
            continue
        seen_positions.add(key)
        unique_atoms.append(int(atom_number))
        unique_positions.append(list(key))

    return unique_atoms, np.asarray(unique_positions, dtype=np.float32)


def symbols_to_atomic_numbers(symbols: Sequence[str]) -> list[int]:
    from rdkit import Chem

    return [Chem.Atom(symbol).GetAtomicNum() for symbol in symbols]


def is_distance_valid(real_cart: np.ndarray, min_distance: float = 0.1) -> bool:
    if real_cart.shape[0] < 2:
        return True

    distance_matrix = cdist(real_cart, real_cart) + 10 * np.eye(real_cart.shape[0])
    return bool(np.min(distance_matrix) >= float(min_distance))
