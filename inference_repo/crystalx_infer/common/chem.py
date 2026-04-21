"""Chemistry helpers shared across CrystalX inference workflows."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def _require_rdkit_chem():
    from rdkit import Chem

    return Chem


def atomic_num_from_symbol(symbol: str) -> int:
    """Return the atomic number for a chemical symbol."""

    return int(_require_rdkit_chem().Atom(symbol.capitalize()).GetAtomicNum())


def atomic_num_list(symbols: Iterable[str]) -> list[int]:
    """Convert an iterable of element symbols into atomic numbers."""

    return [atomic_num_from_symbol(symbol) for symbol in symbols]


def atomic_symbol_from_z(atomic_num: int) -> str:
    """Return an element symbol for an atomic number."""

    chem = _require_rdkit_chem()
    periodic_table = chem.GetPeriodicTable()
    try:
        return str(periodic_table.GetElementSymbol(int(atomic_num)))
    except Exception:
        return f"Z{int(atomic_num)}"


def build_formula(atom_symbols: Iterable[str]) -> str:
    """Build a compact element-count formula preserving first-seen order."""

    counter = Counter(atom_symbols)
    return "".join(f"{symbol}{count}" for symbol, count in counter.items())
