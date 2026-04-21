from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, TextIO

import json
import os
import random

import numpy as np
import torch


DEFAULT_TIMEZONE = "Asia/Singapore"


@dataclass(frozen=True)
class EvalMetrics:
    atom_accuracy: float
    mol_accuracy: float
    total_atoms: int
    total_mols: int
    skipped_mols: int = 0


def set_seed(seed: int = 42, deterministic_cudnn: bool = False) -> None:
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


def get_run_timestamp(timezone_name: str = DEFAULT_TIMEZONE) -> str:
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo(timezone_name))
    except Exception:
        now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)

    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    if isinstance(value, (Path, torch.device)):
        return str(value)
    return value


def write_hparams(log_f: TextIO, hparams: dict[str, Any]) -> None:
    log_f.write("---- Configuration ----\n")
    log_f.write(json.dumps(to_serializable(hparams), indent=2, ensure_ascii=False) + "\n")
    log_f.write("-----------------------\n\n")


def write_log_header(
    log_f: TextIO,
    *,
    run_ts: str,
    device: torch.device,
    test_years: Iterable[int],
    train_size: int,
    test_size: int,
) -> None:
    log_f.write("==== New Run ====\n")
    log_f.write(f"Run timestamp: {run_ts}\n")
    log_f.write(f"Device: {device}\n\n")
    log_f.write(f"Test years: {list(test_years)}\n")
    log_f.write("Train years: all other years\n")
    log_f.write(f"Train Data: {train_size} | Test Data: {test_size}\n\n")


def preview_missing_files(missing: list[str], limit: int = 10) -> None:
    if not missing:
        return

    print(f"[WARN] missing pt files: {len(missing)}")
    for path in missing[:limit]:
        print("  ", path)
