"""Phase 9 ablation: how much does ensemble size K matter?

No retraining is needed -- we re-extract features using subsets of the already-trained
members, so K=1..5 is cheap. K=1 is the important case: it removes ensemble disagreement
entirely, leaving only single-model entropy and latent signals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.experiment import CONFIGS  # noqa: E402
from src.data.collect import Trajectories  # noqa: E402
from src.evaluation.analysis import run_analysis  # noqa: E402
from src.models.train import load_member  # noqa: E402
from src.utils.seeding import get_device  # noqa: E402
from scripts.run_experiment import add_latent_nn_distance, extract_split  # noqa: E402

SPLITS = ["A_gtrain", "A_val", "A_test", "B_test", "C_test", "D_test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 2, 3, 5])
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    device = get_device()
    root = ROOT / "results" / cfg.name
    ab = root / "ablation_K"
    ab.mkdir(parents=True, exist_ok=True)
    data = {k: Trajectories.load(ROOT / "experiments" / cfg.name / f"traj_{k}.npz")
            for k in SPLITS}
    all_models = [load_member(ROOT / "checkpoints" / cfg.name /
                              f"wm_seed{args.seed}_member{k}.pt", device)
                  for k in range(cfg.ensemble_k)]

    summary = {}
    for K in args.ks:
        if K > len(all_models):
            continue
        models = all_models[:K]
        splits = {k: extract_split(models, data[k], cfg, device, f"K{K}:{k}") for k in SPLITS}
        add_latent_nn_distance(splits)
        payload = {f"{k}__{f}": v[f] for k, v in splits.items()
                   for f in ["X", "y", "y_nonmono", "y_m0", "E", "groups", "starts",
                             "priv", "acts"]}
        payload["names"] = np.array(splits["A_test"]["names"], dtype=object)
        fp = ab / f"features_seed{args.seed}_K{K}.npz"
        np.savez_compressed(fp, **payload)
        gm = {k: list(map(int, v)) for k, v in splits["A_test"]["feature_groups"].items()}
        r = run_analysis(fp, gm, ab, seed=args.seed, device=str(device), cfg=cfg,
                         n_boot=max(500, cfg.n_boot // 4))
        prim = r["primary_variant"]
        summary[K] = {
            "primary": prim,
            "mae": r["metrics"][f"g_{prim}"]["mae"]["point"],
            "spearman": r["metrics"][f"g_{prim}"]["spearman"]["point"],
            "constant_mae": r["metrics"]["B1_fixed_horizon"]["mae"]["point"],
            "ceiling_mae": r["ceiling_metrics"][r["best_ceiling"]]["mae"]["point"],
            "best_ceiling": r["best_ceiling"],
            "H_star_mean": r["degeneracy"]["A_test"]["mean"],
        }
        print(f"K={K}: primary={prim} MAE={summary[K]['mae']:.3f} "
              f"rho={summary[K]['spearman']:.3f} const={summary[K]['constant_mae']:.3f} "
              f"ceiling={summary[K]['ceiling_mae']:.3f}", flush=True)
        fp.unlink()  # features are large; the analysis artefacts are what we keep

    json.dump(summary, open(root / f"ablation_K_seed{args.seed}.json", "w"), indent=2)
    print("\nsaved", root / f"ablation_K_seed{args.seed}.json")


if __name__ == "__main__":
    main()
