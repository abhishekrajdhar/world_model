"""Metric verification: decompose H* into its causes before trusting any downstream result."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.experiment import CONFIGS  # noqa: E402
from src.data.collect import Trajectories  # noqa: E402
from src.data.dataset import gather, window_starts  # noqa: E402
from src.envs.configs import A_FORWARD  # noqa: E402
from src.horizon.rih import CONTEXT, EPSILON, H_MAX, error_curve, rih_from_errors, rollout  # noqa: E402
from src.models.train import load_member  # noqa: E402
from src.models.world_model import integrate_displacement  # noqa: E402
from src.utils.seeding import get_device  # noqa: E402


def main(config="pilot", seed=0, split="A_test", n=1024):
    cfg = CONFIGS[config]
    device = get_device()
    traj = Trajectories.load(ROOT / "experiments" / cfg.name / f"traj_{split}.npz")
    models = [load_member(ROOT / "checkpoints" / cfg.name / f"wm_seed{seed}_member{k}.pt", device)
              for k in range(cfg.ensemble_k)]

    L = CONTEXT + H_MAX + 1
    starts = window_starts(traj, L)[::4][:n]
    b = gather(traj, starts, L, device)
    anchor = CONTEXT - 1

    r = rollout(models, b, CONTEXT, H_MAX)
    true_dir = b["dir"][:, anchor + 1: anchor + H_MAX + 1]
    true_moved = b["moved"][:, anchor + 1: anchor + H_MAX + 1]
    pred_dir = r["p_dir"].argmax(-1)
    pred_moved = r["p_moved"].argmax(-1)

    dir_acc = (pred_dir == true_dir).float().mean(0).cpu().numpy()
    moved_acc = (pred_moved == true_moved).float().mean(0).cpu().numpy()

    E = error_curve(r["disp_ens"], r["disp_true"])
    H = rih_from_errors(E, EPSILON)

    print(f"=== {config} seed{seed} {split}  n={len(starts)} ===")
    print("per-step accuracy of the ENSEMBLE consensus over the imagined rollout:")
    for h in [1, 2, 3, 5, 10, 20, 30]:
        print(f"  h={h:2d}  dir_acc={dir_acc[h-1]:.4f}  moved_acc={moved_acc[h-1]:.4f}  "
              f"E_H mean={E[:, h-1].float().mean():.3f}  P(E_H=0)={(E[:, h-1]==0).float().mean():.3f}")
    print(f"overall dir_acc={dir_acc.mean():.4f} moved_acc={moved_acc.mean():.4f}")
    print(f"H* mean={H.float().mean():.2f} sd={H.float().std():.2f}")

    # which factor breaks first?
    dir_bad = (pred_dir != true_dir).long()
    moved_bad = (pred_moved != true_moved).long()
    first_dir = torch.where(dir_bad.any(1), dir_bad.argmax(1), torch.full_like(H, H_MAX))
    first_moved = torch.where(moved_bad.any(1), moved_bad.argmax(1), torch.full_like(H, H_MAX))
    print(f"first dir error step  (mean) = {first_dir.float().mean():.2f}")
    print(f"first moved error step(mean) = {first_moved.float().mean():.2f}")
    print(f"dir breaks first: {(first_dir < first_moved).float().mean():.3f}  "
          f"moved breaks first: {(first_moved < first_dir).float().mean():.3f}")

    # reference models
    acts = r["actions"]
    dir0 = b["dir"][:, anchor]
    # (a) "no walls" model: perfect turn tracking, forward always succeeds
    dirs, cur = [], dir0.clone()
    moved_nw = []
    for k in range(H_MAX):
        a = acts[:, k]
        cur = torch.where(a == 0, (cur - 1) % 4, torch.where(a == 1, (cur + 1) % 4, cur))
        dirs.append(cur.clone())
        moved_nw.append((a == A_FORWARD).long())
    disp_nw = integrate_displacement(torch.stack(dirs, 1), torch.stack(moved_nw, 1), dir0)
    H_nw = rih_from_errors(error_curve(disp_nw, r["disp_true"]), EPSILON)
    # (b) perfect turns, never moves
    disp_static = integrate_displacement(torch.stack(dirs, 1),
                                         torch.zeros_like(torch.stack(moved_nw, 1)), dir0)
    H_static = rih_from_errors(error_curve(disp_static, r["disp_true"]), EPSILON)
    # (c) oracle dir, learned moved
    disp_odir = integrate_displacement(true_dir, pred_moved, dir0)
    H_odir = rih_from_errors(error_curve(disp_odir, r["disp_true"]), EPSILON)
    # (d) learned dir, oracle moved
    disp_omv = integrate_displacement(pred_dir, true_moved, dir0)
    H_omv = rih_from_errors(error_curve(disp_omv, r["disp_true"]), EPSILON)

    print("\nreference models (mean H*):")
    print(f"  learned world model      {H.float().mean():.2f}")
    print(f"  'no walls' (fwd always)  {H_nw.float().mean():.2f}")
    print(f"  'static' (never moves)   {H_static.float().mean():.2f}")
    print(f"  oracle dir + learned mvd {H_odir.float().mean():.2f}")
    print(f"  learned dir + oracle mvd {H_omv.float().mean():.2f}")

    # how often is a turn action mispredicted (should be ~0: turns are deterministic)
    is_turn = (acts == 0) | (acts == 1)
    print(f"\nfraction of imagined steps where action is a turn: {is_turn.float().mean():.3f}")
    print(f"dir error rate on steps following a turn: "
          f"{(dir_bad.bool() & is_turn).float().sum() / is_turn.float().sum():.4f}")
    fwd = acts == A_FORWARD
    print(f"moved error rate on FORWARD steps: "
          f"{(moved_bad.bool() & fwd).float().sum() / fwd.float().sum():.4f}")
    print(f"moved error rate on NON-forward steps: "
          f"{(moved_bad.bool() & ~fwd).float().sum() / (~fwd).float().sum():.4f}")
    print(f"P(true moved | forward) = {(true_moved.bool() & fwd).float().sum()/fwd.float().sum():.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="A_test")
    a = ap.parse_args()
    main(a.config, a.seed, a.split)
