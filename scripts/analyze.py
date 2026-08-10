"""Run the analysis (Phases 7-10, 13) over saved features, then aggregate across seeds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.experiment import CONFIGS  # noqa: E402
from src.evaluation.analysis import run_analysis  # noqa: E402
from src.utils.seeding import get_device  # noqa: E402


def aggregate(per_seed: list[dict], out: Path):
    """Mean +/- sd across training seeds for the headline metrics."""
    keys = sorted({k for r in per_seed for k in r["metrics"]})
    agg = {}
    for k in keys:
        for metric in ["mae", "rmse", "spearman", "r2"]:
            vals = [r["metrics"][k][metric]["point"] for r in per_seed if k in r["metrics"]]
            agg.setdefault(k, {})[metric] = {
                "mean": float(np.mean(vals)), "sd": float(np.std(vals)), "n_seeds": len(vals)
            }
    dec = {}
    for field in ["vs_constant", "vs_best_ceiling", "vs_hand_actions",
                  "strict_vs_constant", "strict_vs_best_ceiling"]:
        vals = [r["decisions"][field]["point"] for r in per_seed if field in r.get("decisions", {})]
        if vals:
            dec[field] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                          "per_seed": [float(v) for v in vals]}
    sp = [r["decisions"]["spearman"]["point"] for r in per_seed if "spearman" in r.get("decisions", {})]
    payload = {"per_method": agg, "decision_deltas": dec,
               "spearman_full": {"mean": float(np.mean(sp)) if sp else None,
                                 "sd": float(np.std(sp)) if sp else None,
                                 "per_seed": [float(v) for v in sp]}}
    json.dump(payload, open(out, "w"), indent=2)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot", choices=list(CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    seeds = args.seeds if args.seeds else list(cfg.seeds)
    root = ROOT / "results" / cfg.name
    device = get_device()

    per_seed = []
    for s in seeds:
        fp = root / f"features_seed{s}.npz"
        gp = root / f"feature_groups_seed{s}.json"
        if not fp.exists():
            print(f"missing {fp}, skipping")
            continue
        groups_map = {k: list(map(int, v)) for k, v in json.load(open(gp)).items()}
        print(f"--- analysing seed {s} ---", flush=True)
        r = run_analysis(fp, groups_map, root, seed=s, device=str(device), cfg=cfg,
                         n_boot=cfg.n_boot)
        per_seed.append(r)

        d = r["degeneracy"]["A_val"]
        print(f"  A_val degeneracy: iqr={d['iqr']:.1f} ceiling={d['ceiling_frac']:.3f} "
              f"-> degenerate={d['degenerate']}")
        print(f"  primary variant (chosen on val): {r['primary_variant']}")
        print(f"  best Claim-A ceiling: {r['best_ceiling']}")
        for k in ["B1_fixed_horizon", f"g_{r['primary_variant']}", "g_full",
                  "g_strict_apriori", "g_hand_actions_only", "g_uncertainty_only",
                  "B8b_privileged_state", "B8_oracle"]:
            if k in r["metrics"]:
                m = r["metrics"][k]
                print(f"  {k:24s} MAE {m['mae']['point']:.3f} "
                      f"[{m['mae']['lo']:.3f},{m['mae']['hi']:.3f}]  "
                      f"rho {m['spearman']['point']:.3f}  R2 {m['r2']['point']:.3f}")
        cm = r["ceiling_metrics"][r["best_ceiling"]]
        print(f"  {'CEILING(' + r['best_ceiling'] + ')':24s} MAE {cm['mae']['point']:.3f} "
              f"rho {cm['spearman']['point']:.3f}")

    if per_seed:
        payload = aggregate(per_seed, root / "aggregate.json")
        print("\n=== aggregate across seeds ===")
        for k, v in sorted(payload["per_method"].items(), key=lambda kv: kv[1]["mae"]["mean"]):
            print(f"  {k:26s} MAE {v['mae']['mean']:.3f} +/- {v['mae']['sd']:.3f}   "
                  f"rho {v['spearman']['mean']:.3f}")


if __name__ == "__main__":
    main()
