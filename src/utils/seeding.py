"""Deterministic seeding and run provenance."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cuda_usable() -> bool:
    """`torch.cuda.is_available()` is not sufficient: Kaggle sometimes allocates a Pascal
    P100 (sm_60) while the preinstalled torch only ships sm_70+ kernels, so every launch
    fails with cudaErrorNoKernelImageForDevice. Verify with a real kernel launch."""
    if not torch.cuda.is_available():
        return False
    try:
        (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).sum().item()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[warn] CUDA present but unusable ({type(e).__name__}: {e}); falling back")
        return False


def get_device() -> torch.device:
    if cuda_usable():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unavailable"


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False, "device": str(get_device())}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 1024**3, 2),
        "capability": f"{props.major}.{props.minor}",
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }


def provenance(config=None) -> dict:
    import gymnasium
    import minigrid

    return {
        "git_commit": _git_hash(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "minigrid": minigrid.__version__,
        "gpu": gpu_info(),
        "config": (asdict(config) if is_dataclass(config) else config),
    }


def save_provenance(path: str | Path, config=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance(config), indent=2, default=str))
