"""Emit the quantitative tables and the pre-registered verdict from saved analyses."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.experiment import CONFIGS  # noqa: E402

PRETTY = {
    "B1_fixed_horizon": "Baseline 1: fixed horizon (median H*)",
    "B2_random_horizon": "Baseline 2: random horizon",
    "iso_val::context_onestep_error": "Baseline 3: current-step error (isotonic)",
    "iso_val::predictive_entropy_1step": "Baseline 4: model confidence (isotonic)",
    "iso_val::ensemble_disagreement_1step": "Baseline 5: ensemble disagreement (isotonic)",
    "iso_val::latent_prior_std": "Baseline 6a: latent prior std (isotonic)",
    "iso_val::latent_post_prior_kl": "Baseline 6b: latent prior/posterior KL (isotonic)",
    "iso_val::free_ahead": "Baseline 7: visible free cells ahead (isotonic)",
    "iso_val::n_forward_planned": "Baseline 7b: planned forward count (isotonic)",
    "iso_val::internal_truncation_disp": "Online truncation (disp spread >= 1)",
    "iso_val::internal_truncation_js": "Online truncation (JS >= 0.1)",
    "iso_val::imagined_disp_spread": "Imagined displacement spread (isotonic)",
    "iso_val::ensemble_disagreement_mean": "Ensemble disagreement over rollout (isotonic)",
    "B8b_privileged_state": "Baseline 8b: PRIVILEGED true local grid",
    "B8_oracle": "Baseline 8: oracle (uses the future)",
}


def fmt(m, k="mae"):
    v = m[k]
    return f"{v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"


def verdict(per_seed: list[dict]) -> tuple[str, list[str]]:
    """Apply the frozen decision rules from PREREGISTRATION.md sec. 11."""
    notes = []
    # (a) beats constant, non-overlapping CI on the paired difference
    d_const = [r["decisions"]["vs_constant"] for r in per_seed]
    a_ok = all(d["hi"] < 0 for d in d_const)   # negative delta = lower MAE = better
    # (b) beats the best optimistic monotone ceiling
    d_ceil = [r["decisions"]["vs_best_ceiling"] for r in per_seed]
    b_ok = all(d["hi"] < 0 for d in d_ceil)
    # (c) spearman > 0.5 with CI excluding 0.3
    sp = [r["decisions"]["spearman"] for r in per_seed]
    c_ok = all(s["point"] > 0.5 and s["lo"] > 0.3 for s in sp)
    # degeneracy is checked on VALIDATION
    degen = any(r["degeneracy"]["A_val"]["degenerate"] for r in per_seed)

    notes.append(f"(a) beats constant on every seed: {a_ok} "
                 f"(deltas {[round(d['point'],3) for d in d_const]}, "
                 f"upper CIs {[round(d['hi'],3) for d in d_const]})")
    notes.append(f"(b) beats best Claim-A ceiling on every seed: {b_ok} "
                 f"(deltas {[round(d['point'],3) for d in d_ceil]}, "
                 f"upper CIs {[round(d['hi'],3) for d in d_ceil]})")
    notes.append(f"(c) Spearman > 0.5 with CI lower bound > 0.3: {c_ok} "
                 f"(rho {[round(s['point'],3) for s in sp]}, "
                 f"lo {[round(s['lo'],3) for s in sp]})")
    notes.append(f"validation degeneracy triggered: {degen}")

    if degen:
        return "INCONCLUSIVE", notes
    if not a_ok:
        return "NOT SUPPORTED", notes
    if a_ok and b_ok and c_ok:
        return "SUPPORTED", notes
    return "PARTIALLY SUPPORTED", notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    res = Path(args.results) if args.results else ROOT / "results" / cfg.name

    per_seed = []
    for p in sorted(res.glob("analysis_seed*.json")):
        per_seed.append(json.load(open(p)))
    if not per_seed:
        raise SystemExit(f"no analyses found in {res}")
    seeds = [r["seed"] for r in per_seed]
    print(f"# RIH results ({cfg.name}, seeds {seeds})\n")

    d0 = per_seed[0]["degeneracy"]
    print("## H* distribution\n")
    print("| split | mean | sd | IQR | at ceiling | at zero | degenerate |")
    print("|---|---|---|---|---|---|---|")
    for k, v in d0.items():
        print(f"| {k} | {v['mean']:.2f} | {v['sd']:.2f} | {v['iqr']:.1f} | "
              f"{v['ceiling_frac']:.3f} | {v['zero_frac']:.3f} | {v['degenerate']} |")

    primary = per_seed[0]["primary_variant"]
    print(f"\nPrimary g_phi variant (selected on validation): **{primary}**")
    print(f"Best Claim-A ceiling signal: **{per_seed[0]['best_ceiling']}**\n")

    # main table: mean +/- sd across seeds
    keys = sorted({k for r in per_seed for k in r["metrics"]})
    rows = []
    for k in keys:
        vals = {m: [r["metrics"][k][m]["point"] for r in per_seed if k in r["metrics"]]
                for m in ["mae", "rmse", "spearman", "r2"]}
        rows.append((k, vals))
    rows.sort(key=lambda kv: np.mean(kv[1]["mae"]))

    print("## Predicted vs actual RIH on A-test (mean +/- sd over seeds)\n")
    print("| method | MAE | RMSE | Spearman | R2 |")
    print("|---|---|---|---|---|")
    for k, v in rows:
        name = PRETTY.get(k, k.replace("g_", "g_phi: "))
        print(f"| {name} | {np.mean(v['mae']):.3f} ± {np.std(v['mae']):.3f} | "
              f"{np.mean(v['rmse']):.3f} ± {np.std(v['rmse']):.3f} | "
              f"{np.mean(v['spearman']):.3f} ± {np.std(v['spearman']):.3f} | "
              f"{np.mean(v['r2']):.3f} ± {np.std(v['r2']):.3f} |")

    print("\n## Claim-A ceilings (isotonic fitted ON TEST -- deliberately optimistic)\n")
    print("| signal | MAE | Spearman |")
    print("|---|---|---|")
    cm = per_seed[0]["ceiling_metrics"]
    for k in sorted(cm, key=lambda x: cm[x]["mae"]["point"]):
        v = [r["ceiling_metrics"][k] for r in per_seed if k in r["ceiling_metrics"]]
        print(f"| {k} | {np.mean([x['mae']['point'] for x in v]):.3f} | "
              f"{np.mean([x['spearman']['point'] for x in v]):.3f} |")

    print("\n## Claim A: uncertainty vs future error (Spearman, A-test)\n")
    print("| signal | vs E_1 | vs E_5 | vs E_10 | vs E_20 | vs H* |")
    print("|---|---|---|---|---|---|")
    ca = per_seed[0]["claim_A"]
    for k in ca:
        v = [r["claim_A"][k] for r in per_seed]
        cells = [np.mean([x[f"spearman_vs_E{h}"] for x in v]) for h in [1, 5, 10, 20]]
        hs = np.mean([x["spearman_vs_Hstar"] for x in v])
        print(f"| {k} | " + " | ".join(f"{c:+.3f}" for c in cells) + f" | {hs:+.3f} |")

    print("\n## H3: transfer under distribution shift\n")
    print("| split | actual H* | predicted H* | MAE | Spearman |")
    print("|---|---|---|---|---|")
    print(f"| A_test (in-dist) | {d0['A_test']['mean']:.2f} | "
          f"{np.mean([r['metrics'][f'g_{primary}']['mae']['point'] for r in per_seed]):.3f} MAE | "
          f"- | {np.mean([r['decisions']['spearman']['point'] for r in per_seed]):.3f} |")
    for key in ["B_test", "C_test", "D_test"]:
        v = [r["ood"][key] for r in per_seed if key in r.get("ood", {})]
        if not v or primary not in v[0]:
            continue
        print(f"| {key} | {np.mean([x['H_star']['mean'] for x in v]):.2f} | "
              f"{np.mean([x[primary]['pred_mean'] for x in v]):.2f} | "
              f"{np.mean([x[primary]['mae']['point'] for x in v]):.3f} | "
              f"{np.mean([x[primary]['spearman']['point'] for x in v]):.3f} |")

    v, notes = verdict(per_seed)
    print("\n## Pre-registered verdict\n")
    for n in notes:
        print(f"- {n}")
    print(f"\n### **{v}**")

    pl = res / "planning.json"
    if pl.exists():
        P = json.load(open(pl))
        print("\n## H4: adaptive vs fixed planning\n")
        print("| arm | success | steps to goal | imagined states / decision | mean H |")
        print("|---|---|---|---|---|")
        for k in [x for x in P if x != "comparison"]:
            r = P[k]
            print(f"| {k} | {r['success_rate']:.3f} ± {r['success_se']:.3f} | "
                  f"{r['steps_to_goal_mean']:.1f} | {r['imagined_states_mean']:.0f} | "
                  f"{r['mean_horizon']:.1f} |")
        c = P["comparison"]
        print(f"\nadaptive vs {c['best_fixed']}: success delta "
              f"{c['success_delta']['point']:+.3f} "
              f"[{c['success_delta']['lo']:+.3f}, {c['success_delta']['hi']:+.3f}], "
              f"compute ratio {c['compute_ratio']:.2f}x")


if __name__ == "__main__":
    main()
