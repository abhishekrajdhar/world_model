"""Phase 11: adaptive-horizon planning vs fixed horizons (paired on identical env seeds)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.experiment import CONFIGS  # noqa: E402
from src.envs.configs import ENV_CONFIGS  # noqa: E402
from src.evaluation.analysis import VARIANTS, cols_for, load_features  # noqa: E402
from src.evaluation.stats import bootstrap_ci  # noqa: E402
from src.horizon.predictor import train_predictor  # noqa: E402
from src.models.train import load_member  # noqa: E402
from src.planning.adaptive import run_episode  # noqa: E402
from src.utils.seeding import get_device, set_seed  # noqa: E402

FIXED = [3, 5, 10, 20, 30]


def summarise(rows: list[dict]) -> dict:
    rows = [r for r in rows if r]
    ok = np.array([r["reached"] for r in rows], float)
    steps = np.array([r["steps"] for r in rows], float)
    imag = np.array([r["imagined_per_decision"] for r in rows], float)
    dist = np.array([r["final_dist"] for r in rows], float)
    return {
        "n": len(rows),
        "mean_detour": float(np.mean([r["detour"] for r in rows])) if rows else float("nan"),
        "mean_path_dist": float(np.mean([r["path_dist"] for r in rows])) if rows else float("nan"),
        "tier_counts": {str(t): int(sum(1 for r in rows if r["tier"] == t)) for t in range(4)},
        "success_rate": float(ok.mean()),
        "success_se": float(ok.std() / np.sqrt(len(ok))),
        "steps_to_goal_mean": float(steps.mean()),
        "steps_se": float(steps.std() / np.sqrt(len(steps))),
        "imagined_states_mean": float(imag.mean()),
        "final_dist_mean": float(dist.mean()),
        "mean_horizon": float(np.mean([r["mean_horizon"] for r in rows])),
        "_reached": ok.tolist(), "_imag": imag.tolist(), "_dist": dist.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--n-samples", type=int, default=48)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--results", default=None)
    ap.add_argument("--fixed", type=int, nargs="*", default=None,
                    help="fixed horizons to evaluate (default 3 5 10 20 30)")
    args = ap.parse_args()
    fixed = args.fixed if args.fixed else FIXED

    cfg = CONFIGS[args.config]
    device = get_device()
    res = Path(args.results) if args.results else ROOT / "results" / cfg.name
    set_seed(args.seed)

    models = [load_member(ROOT / "checkpoints" / cfg.name /
                          f"wm_seed{args.seed}_member{k}.pt", device)
              for k in range(cfg.ensemble_k)]

    # refit the primary predictor so the planner uses exactly the reported model
    ana = json.load(open(res / f"analysis_seed{args.seed}.json"))
    primary = ana["primary_variant"]
    groups_map = {k: list(map(int, v)) for k, v in
                  json.load(open(res / f"feature_groups_seed{args.seed}.json")).items()}
    data = load_features(res / f"features_seed{args.seed}.npz")
    S = data["splits"]
    cols = cols_for(groups_map, VARIANTS[primary], S["A_test"]["X"].shape[1])
    pred_model, std, _ = train_predictor(
        S["A_gtrain"]["X"][:, cols], S["A_gtrain"]["y"],
        S["A_val"]["X"][:, cols], S["A_val"]["y"],
        seed=args.seed, epochs=cfg.g_epochs, hidden=cfg.g_hidden, depth=cfg.g_depth,
        lr=cfg.g_lr, device=str(device))
    print(f"planner uses g_phi variant '{primary}' ({len(cols)} features)", flush=True)

    env_cfg = ENV_CONFIGS["A"]
    # paired: every arm sees the SAME layout seeds and the same planner RNG seed
    layout_seeds = list(range(90000, 90000 + args.episodes))
    raw: dict[str, list] = {}
    out: dict = {}
    for arm in [*fixed, "adaptive"]:
        key = f"fixed_{arm}" if arm != "adaptive" else "adaptive"
        t0 = time.time()
        raw[key] = [run_episode(
            models, env_cfg, ls, arm, n_samples=args.n_samples, budget=args.budget,
            device=device, predictor=pred_model, standardizer=std, cols=cols,
            rng_seed=1000 + i) for i, ls in enumerate(layout_seeds)]
        out[key] = {"wallclock_s": time.time() - t0}

    # keep only layouts where EVERY arm produced an episode, so comparisons stay paired
    valid = [i for i in range(len(layout_seeds))
             if all(raw[k][i] is not None for k in raw)]
    print(f"{len(valid)}/{len(layout_seeds)} layouts usable (detour target found)")
    for key in raw:
        wc = out[key]["wallclock_s"]
        out[key] = summarise([raw[key][i] for i in valid])
        out[key]["wallclock_s"] = wc
        print(f"  {key:12s} success {out[key]['success_rate']:.3f} "
              f"steps {out[key]['steps_to_goal_mean']:.1f} "
              f"imag/decision {out[key]['imagined_states_mean']:.0f} "
              f"H {out[key]['mean_horizon']:.1f} "
              f"detour {out[key]['mean_detour']:.1f} path {out[key]['mean_path_dist']:.1f} "
              f"tiers {out[key]['tier_counts']} ({out[key]['wallclock_s']:.0f}s)", flush=True)

    # paired comparison: adaptive vs the best fixed arm (best by success rate)
    best_fixed = max((k for k in out if k.startswith("fixed_")),
                     key=lambda k: out[k]["success_rate"])
    g = np.arange(len(valid))
    a = np.array(out["adaptive"]["_reached"])
    b = np.array(out[best_fixed]["_reached"])
    ai = np.array(out["adaptive"]["_imag"])
    bi = np.array(out[best_fixed]["_imag"])
    out["comparison"] = {
        "best_fixed": best_fixed,
        "success_delta": bootstrap_ci(lambda x, y: float((x - y).mean()), g, a, b,
                                      n_boot=2000, seed=0),
        "compute_ratio": float(ai.mean() / max(1e-9, bi.mean())),
        "compute_delta": bootstrap_ci(lambda x, y: float((x - y).mean()), g, ai, bi,
                                      n_boot=2000, seed=0),
    }
    json.dump(out, open(res / "planning.json", "w"), indent=2)
    c = out["comparison"]
    print(f"\nadaptive vs {best_fixed}: success delta {c['success_delta']['point']:+.3f} "
          f"[{c['success_delta']['lo']:+.3f},{c['success_delta']['hi']:+.3f}]  "
          f"compute ratio {c['compute_ratio']:.2f}x")
    print("saved", res / "planning.json")


if __name__ == "__main__":
    main()
