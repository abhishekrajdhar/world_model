"""Run the H6 transfer-matrix experiment across seeds and emit the verdict."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.experiment import CONFIGS as EXP_CONFIGS  # noqa: E402
from src.evaluation.analysis import load_features  # noqa: E402
from src.horizon.transfer import CONFIGS, REGIMES, evaluate_criteria, run_transfer  # noqa: E402
from src.utils.seeding import get_device  # noqa: E402


def mean_over(rs, path):
    vals = []
    for r in rs:
        v = r
        for k in path:
            v = v[k]
        vals.append(v)
    return float(np.mean(vals))


def report(per_seed, tag):
    print(f"\n{'='*72}\n H6 {tag}  (mean over {len(per_seed)} seeds)\n{'='*72}")
    print("\ntraining anchors per regime:",
          {k: int(np.mean([r["n_train_per_regime"][k] for r in per_seed]))
           for k in per_seed[0]["n_train_per_regime"]})
    print("test anchors per config:", per_seed[0]["n_test_per_config"])

    print("\n### Predicted vs actual H* (bold = config unseen in training)\n")
    print("| regime | " + " | ".join(f"test {c}" for c in CONFIGS) + " |")
    print("|---|" + "---|" * len(CONFIGS))
    act = {c: mean_over(per_seed, ["metrics", "A", c, "actual_mean"]) for c in CONFIGS}
    print("| *actual* | " + " | ".join(f"{act[c]:.2f}" for c in CONFIGS) + " |")
    for rname in REGIMES:
        cells = []
        for c in CONFIGS:
            p = mean_over(per_seed, ["metrics", rname, c, "pred_mean"])
            seen = c in REGIMES[rname]
            cells.append(f"{p:.2f}" if seen else f"**{p:.2f}**")
        print(f"| {rname} | " + " | ".join(cells) + " |")

    for metric, fmt in [("mae", "{:.3f}"), ("spearman", "{:+.3f}")]:
        print(f"\n### {metric.upper()} (bold = unseen)\n")
        print("| regime | " + " | ".join(f"test {c}" for c in CONFIGS) + " |")
        print("|---|" + "---|" * len(CONFIGS))
        for rname in REGIMES:
            cells = []
            for c in CONFIGS:
                v = mean_over(per_seed, ["metrics", rname, c, metric, "point"])
                s = fmt.format(v)
                cells.append(s if c in REGIMES[rname] else f"**{s}**")
            print(f"| {rname} | " + " | ".join(cells) + " |")

    print("\n### Transfer T_MAE(regime -> config) = MAE(g_A) - MAE(g_regime); "
          "positive = improvement over IID-only\n")
    print("| regime | " + " | ".join(f"test {c}" for c in CONFIGS) + " |")
    print("|---|" + "---|" * len(CONFIGS))
    for rname in REGIMES:
        if rname == "A":
            continue
        cells = []
        for c in CONFIGS:
            v = mean_over(per_seed, ["transfer", rname, c, "T_MAE", "point"])
            lo = mean_over(per_seed, ["transfer", rname, c, "T_MAE", "lo"])
            hi = mean_over(per_seed, ["transfer", rname, c, "T_MAE", "hi"])
            s = f"{v:+.3f} [{lo:+.2f},{hi:+.2f}]"
            cells.append(s if c in REGIMES[rname] else f"**{s}**")
        print(f"| {rname} | " + " | ".join(cells) + " |")

    print("\n### Rank stability of uncertainty features\n")
    print("| feature | rho_A | rho_B | rho_C | rho_D | mean|rho_OOD| | RS = Var_D[rho] |")
    print("|---|---|---|---|---|---|---|")
    rs = per_seed[0]["rank_stability"]
    rows = []
    for label in rs:
        rhos = {c: mean_over(per_seed, ["rank_stability", label, "rho", c]) for c in CONFIGS}
        v = mean_over(per_seed, ["rank_stability", label, "rank_stability_var"])
        mo = mean_over(per_seed, ["rank_stability", label, "mean_abs_rho_ood"])
        rows.append((label, rhos, mo, v))
    for label, rhos, mo, v in sorted(rows, key=lambda r: -abs(r[1]["A"])):
        print(f"| {label} | " + " | ".join(f"{rhos[c]:+.3f}" for c in CONFIGS) +
              f" | {mo:.3f} | {v:.4f} |")

    print("\n### Criteria on completely held-out cells\n")
    verdicts = []
    for r in per_seed:
        crit = evaluate_criteria(r)
        verdicts.append(crit["verdict"])
        r["criteria"] = crit
    for key in sorted(per_seed[0]["criteria"]["cells"]):
        flags = [r["criteria"]["cells"][key] for r in per_seed]
        restored = sum(f["restored"] for f in flags)
        c1 = sum(f["c1_more_conservative_than_ID"] for f in flags)
        c2 = sum(f["c2_spearman_gt_0.3"] for f in flags)
        c3 = sum(f["c3_beats_gA_MAE"] for f in flags)
        print(f"  {key:14s} c1 {c1}/{len(flags)}  c2 {c2}/{len(flags)}  c3 {c3}/{len(flags)}"
              f"  -> restored on {restored}/{len(flags)} seeds")
    print("\n  in-distribution preserved (A-test MAE within 0.5 of A-only):",
          {r: all(x["criteria"]["id_preserved"][r] for x in per_seed) for r in REGIMES})
    overall = ("SUPPORTED" if all(v == "SUPPORTED" for v in verdicts) else
               "PARTIALLY SUPPORTED" if all(v != "NOT SUPPORTED" for v in verdicts) else
               "NOT SUPPORTED")
    print(f"\n### H6 verdict ({tag}): **{overall}**   per seed: {verdicts}")
    return overall, verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    cfg = EXP_CONFIGS[args.config]
    res = Path(args.results) if args.results else ROOT / "results" / cfg.name
    out_dir = ROOT / "results" / "transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = str(get_device())
    print("device:", device, flush=True)

    all_runs = {}
    for matched in (False, True):
        tag = "size-matched control" if matched else "main"
        per_seed = []
        for s in args.seeds:
            fp = res / f"features_seed{s}.npz"
            gp = res / f"feature_groups_seed{s}.json"
            if not fp.exists():
                print(f"missing {fp}; skipping")
                continue
            data = load_features(fp)
            gm = {k: list(map(int, v)) for k, v in json.load(open(gp)).items()}
            print(f"--- {tag}: seed {s} ---", flush=True)
            r = run_transfer(data["splits"], data["names"], gm, seed=s, cfg_obj=cfg,
                             device=device, n_boot=cfg.n_boot, size_matched=matched)
            per_seed.append(r)
        if not per_seed:
            raise SystemExit("no seeds processed")
        overall, verdicts = report(per_seed, tag)
        all_runs[tag] = {"overall": overall, "verdicts": verdicts}
        json.dump(per_seed, open(out_dir / f"transfer_{'matched' if matched else 'main'}.json",
                                 "w"), indent=2, default=float)

    json.dump(all_runs, open(out_dir / "transfer_summary.json", "w"), indent=2)
    print("\n\n### FINAL: main =", all_runs["main"]["overall"],
          "| size-matched control =", all_runs["size-matched control"]["overall"])


if __name__ == "__main__":
    main()
