"""Runtime helpers for CrystalX inference entrypoints."""

from __future__ import annotations

from datetime import datetime
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cudnn: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible evaluation."""

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


def get_run_timestamp(timezone_name: str = "Asia/Singapore") -> str:
    """Return a filesystem-safe timestamp string."""

    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo(timezone_name))
    except Exception:
        now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def resolve_device(device_name: str = "auto") -> torch.device:
    """Resolve the requested device string into a torch.device."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def unwrap_state_dict(checkpoint):
    """Unwrap a training checkpoint into a plain model state dict."""

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint
