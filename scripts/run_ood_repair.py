"""Run the H5 OOD-repair experiment over all training seeds and emit the verdict."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.experiment import CONFIGS  # noqa: E402
from src.evaluation.analysis import load_features  # noqa: E402
from src.horizon.ood_repair import OOD_SPLITS, evaluate_criteria, run_arms  # noqa: E402
from src.utils.seeding import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    res = Path(args.results) if args.results else ROOT / "results" / cfg.name
    out_dir = ROOT / "results" / "ood_repair"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device={device}", flush=True)

    per_seed = []
    for s in args.seeds:
        fp = res / f"features_seed{s}.npz"
        gp = res / f"feature_groups_seed{s}.json"
        if not fp.exists():
            print(f"missing {fp}; skipping seed {s}")
            continue
        data = load_features(fp)
        groups_map = {k: list(map(int, v)) for k, v in json.load(open(gp)).items()}
        print(f"--- seed {s} ---", flush=True)
        r = run_arms(data["splits"], data["names"], groups_map, seed=s, cfg=cfg,
                     device=str(device), n_boot=cfg.n_boot)
        r["criteria"] = evaluate_criteria(r["arms"])
        per_seed.append(r)
        json.dump(r, open(out_dir / f"ood_repair_seed{s}.json", "w"), indent=2, default=float)
        for arm, rec in r["arms"].items():
            print(f"  {arm:14s} A_test pred {rec['A_test']['pred_mean']:5.2f} "
                  f"MAE {rec['A_test']['mae']['point']:.3f} | " +
                  " ".join(f"{sp[0]} pred {rec[sp]['pred_mean']:5.2f} "
                           f"rho {rec[sp]['spearman']['point']:+.3f}" for sp in OOD_SPLITS),
                  flush=True)
        print(f"  verdict(seed {s}): {r['criteria']['verdict']}", flush=True)

    if not per_seed:
        raise SystemExit("no seeds processed")

    # ---- aggregate ----
    print("\n=== H5 diagnostic: uncertainty signals per split (mean over seeds) ===")
    print("| feature | " + " | ".join(["A_test", *OOD_SPLITS]) + " |")
    print("|---|" + "---|" * 4)
    for feat in per_seed[0]["diagnostic"]:
        cells = []
        for sp in ["A_test", *OOD_SPLITS]:
            m = np.mean([r["diagnostic"][feat][sp]["mean"] for r in per_seed])
            rho = np.mean([r["diagnostic"][feat][sp]["rho_with_Hstar_within_split"]
                           for r in per_seed])
            cells.append(f"{m:.3f} (rho {rho:+.2f})")
        print(f"| {feat} | " + " | ".join(cells) + " |")

    print("\n=== H5 arms (mean over seeds) ===")
    print("| arm | split | actual H* | predicted H* | MAE | Spearman |")
    print("|---|---|---|---|---|---|")
    for arm in per_seed[0]["arms"]:
        for sp in ["A_test", *OOD_SPLITS]:
            g = lambda k, f=None: np.mean([  # noqa: E731
                (r["arms"][arm][sp][k] if f is None else r["arms"][arm][sp][k][f])
                for r in per_seed])
            print(f"| {arm} | {sp} | {g('actual_mean'):.2f} | {g('pred_mean'):.2f} | "
                  f"{g('mae','point'):.3f} | {g('spearman','point'):+.3f} |")

    verdicts = [r["criteria"]["verdict"] for r in per_seed]
    agg = {"per_seed_verdicts": verdicts,
           "arms_supporting": [r["criteria"]["arms_supporting"] for r in per_seed],
           "arms_partial": [r["criteria"]["arms_partial"] for r in per_seed]}
    # overall = the verdict holding on every seed
    overall = ("SUPPORTED" if all(v == "SUPPORTED" for v in verdicts) else
               "PARTIALLY SUPPORTED" if all(v != "NOT SUPPORTED" for v in verdicts) else
               "NOT SUPPORTED")
    agg["overall"] = overall
    print("\n=== per-seed criteria ===")
    for r in per_seed:
        for arm in ["R1_ranknorm", "R2_gated", "R3_combined"]:
            c = r["criteria"][arm]
            print(f"  seed {r['seed']} {arm:14s} c1={c['c1_horizon_falls_on_all_ood']} "
                  f"c2={c['c2_spearman_gt_0.3_on_2of3']} c3={c['c3_id_mae_not_degraded']} "
                  f"-> {'SUPPORTED' if c['supported'] else 'partial' if c['partial'] else 'no'}")
    print(f"\n### H5 verdict: **{overall}**  (per seed: {verdicts})")
    json.dump(agg, open(out_dir / "ood_repair_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
