from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class SplitSpec:
    """Dataset split configuration based on year metadata and pt file layout."""

    txt_path: str
    pt_dir: str
    test_years: tuple[int, ...] = (2022, 2023, 2024)
    pt_prefix: str = "equiv_"
    pt_suffix: str = ".pt"
    strict: bool = False

    def normalized_test_years(self) -> set[str]:
        return {str(year) for year in self.test_years}


def split_by_year_txt(spec: SplitSpec) -> tuple[list[str], list[str], list[str]]:
    test_years = spec.normalized_test_years()
    pt_dir = Path(spec.pt_dir)

    train_files: list[str] = []
    test_files: list[str] = []
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

            if year in test_years:
                test_files.append(pt_path)
            else:
                train_files.append(pt_path)

    if not spec.strict and pt_dir.exists():
        listed_files = set(train_files) | set(test_files)
        extra_train_files = sorted(
            str(path)
            for path in pt_dir.iterdir()
            if path.is_file()
            and path.name.endswith(spec.pt_suffix)
            and str(path) not in listed_files
        )
        train_files.extend(extra_train_files)

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
