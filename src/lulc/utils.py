"""Small shared helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def get_device() -> torch.device:
    """CUDA when present, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory (and parents) and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def free_cuda() -> None:
    """Release cached GPU blocks between XAI passes."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
