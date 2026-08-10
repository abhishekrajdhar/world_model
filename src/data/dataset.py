"""Sequence sampling over collected trajectories."""
from __future__ import annotations

import numpy as np
import torch

from src.data.collect import Trajectories

KEYS = ["image", "dir", "carrying", "door_open", "moved", "action"]


def window_starts(traj: Trajectories, length: int) -> np.ndarray:
    """Global indices at which a contiguous window of `length` fits inside one episode."""
    outs = []
    for s, l in zip(traj.ep_start, traj.ep_len):
        if l >= length:
            outs.append(np.arange(s, s + l - length + 1))
    return np.concatenate(outs) if outs else np.zeros(0, dtype=np.int64)


def anchor_indices(traj: Trajectories, context: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Anchors t with `context` observed steps before/at t and `horizon` real steps after.

    Returns (window_start, anchor_offset_within_window) with window length context+horizon.
    """
    starts = window_starts(traj, context + horizon)
    return starts, np.full(len(starts), context - 1, dtype=np.int64)


def episode_of(traj: Trajectories, idx: np.ndarray) -> np.ndarray:
    """Episode id for each global index (used for clustered bootstrap)."""
    return np.searchsorted(traj.ep_start, idx, side="right") - 1


def gather(traj: Trajectories, starts: np.ndarray, length: int, device="cpu") -> dict:
    off = np.arange(length)[None, :]
    idx = starts[:, None] + off
    out = {
        "image": torch.as_tensor(traj.image[idx].astype(np.int64)),
        "dir": torch.as_tensor(traj.dir[idx].astype(np.int64)),
        "carrying": torch.as_tensor(traj.carrying[idx].astype(np.int64)),
        "door_open": torch.as_tensor(traj.door_open[idx].astype(np.int64)),
        "moved": torch.as_tensor(traj.moved[idx].astype(np.int64)),
        "action": torch.as_tensor(traj.action[idx].astype(np.int64)),
        "pos": torch.as_tensor(traj.pos[idx].astype(np.int64)),
        # privileged: true local grid. Never given to g_phi; used only for the upper bound.
        "priv": torch.as_tensor(traj.priv[idx].astype(np.int64)),
    }
    return {k: v.to(device) for k, v in out.items()}


class SequenceSampler:
    def __init__(self, traj: Trajectories, length: int, batch_size: int, seed: int = 0):
        self.traj, self.length, self.batch_size = traj, length, batch_size
        self.starts = window_starts(traj, length)
        if len(self.starts) == 0:
            raise ValueError("no episode long enough for the requested sequence length")
        self.rng = np.random.default_rng(seed)

    def sample(self, device="cpu") -> dict:
        sel = self.rng.choice(self.starts, size=self.batch_size, replace=True)
        return gather(self.traj, sel, self.length, device)
