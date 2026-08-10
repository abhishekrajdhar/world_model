"""Verify the information boundary: g_phi's features must not depend on the real future.

Method: corrupt everything in the window after the anchor EXCEPT the planned action
sequence (which a planner legitimately knows), then check that
  (a) every feature is bit-identical, and
  (b) the ground-truth H* DOES change -- otherwise the target is not actually reading the
      future and the whole experiment would be measuring nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.experiment import CONFIGS  # noqa: E402
from src.data.collect import Trajectories  # noqa: E402
from src.data.dataset import gather, window_starts  # noqa: E402
from src.horizon.rih import CONTEXT, H_MAX, compute_ground_truth  # noqa: E402
from src.models.train import load_member  # noqa: E402
from src.uncertainty.signals import extract_features  # noqa: E402
from src.utils.seeding import get_device  # noqa: E402

FUTURE_KEYS = ["image", "dir", "carrying", "door_open", "moved", "pos", "priv"]


def build(config="pilot", seed=0, n=256):
    cfg = CONFIGS[config]
    device = get_device()
    traj = Trajectories.load(ROOT / "experiments" / cfg.name / "traj_A_test.npz")
    models = [load_member(ROOT / "checkpoints" / cfg.name / f"wm_seed{seed}_member{k}.pt", device)
              for k in range(cfg.ensemble_k)]
    L = CONTEXT + H_MAX + 1
    starts = window_starts(traj, L)[:n]
    return models, gather(traj, starts, L, device), device


def corrupt_future(batch, rng_seed=0):
    """Randomise every future field except `action`."""
    g = torch.Generator(device="cpu").manual_seed(rng_seed)
    out = {k: v.clone() for k, v in batch.items()}
    a = CONTEXT  # first strictly-future index
    for k in FUTURE_KEYS:
        if k not in out:
            continue
        seg = out[k][:, a:]
        hi = int(seg.max().item()) + 1 if seg.numel() else 2
        out[k][:, a:] = torch.randint(0, max(2, hi), seg.shape, generator=g).to(seg.device)
    return out


def main():
    models, batch, device = build()
    r_clean = compute_ground_truth(models, batch, CONTEXT, H_MAX)
    X_clean, names, _ = extract_features(models, batch, r_clean["rollout"], CONTEXT, H_MAX)

    bad = corrupt_future(batch)
    r_bad = compute_ground_truth(models, bad, CONTEXT, H_MAX)
    X_bad, _, _ = extract_features(models, bad, r_bad["rollout"], CONTEXT, H_MAX)

    same = np.array_equal(X_clean, X_bad)
    diff_cols = [names[i] for i in np.where(~np.isclose(X_clean, X_bad, atol=0, rtol=0).all(0))[0]]
    h_clean = r_clean["H_star"].cpu().numpy()
    h_bad = r_bad["H_star"].cpu().numpy()
    target_changed = not np.array_equal(h_clean, h_bad)

    print(f"features identical after corrupting the future : {same}")
    if not same:
        print(f"  LEAKING COLUMNS ({len(diff_cols)}): {diff_cols[:20]}")
    print(f"ground-truth H* changed after corruption       : {target_changed} "
          f"(mean {h_clean.mean():.2f} -> {h_bad.mean():.2f})")

    # actions must be preserved by the corruption, else the test is vacuous
    assert torch.equal(batch["action"], bad["action"]), "action sequence must be preserved"
    assert same, "FEATURE LEAKAGE DETECTED: a feature depends on the real future"
    assert target_changed, "H* does not depend on the future -- target is broken"
    print("\nPASS: no future information reaches g_phi, and the target does read the future.")


if __name__ == "__main__":
    main()
