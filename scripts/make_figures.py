"""Generate all publication figures from saved analysis artefacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.experiment import CONFIGS  # noqa: E402
from src.evaluation import figures as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default=None, help="override results dir")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    res = Path(args.results) if args.results else ROOT / "results" / cfg.name
    out = ROOT / "figures" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    s = args.seed

    d = np.load(res / f"predictions_seed{s}.npz")
    ana = json.load(open(res / f"analysis_seed{s}.json"))
    primary = ana.get("primary_variant", "full")
    y = d["y_test"]
    pred = d[f"pred__g_{primary}"]

    F.fig1_architecture(out)

    # Fig 2: error vs horizon, in-distribution and under shift
    E_by = {"A (train dist.)": d["E_test"]}
    for k, lab in [("B_test", "B (unseen layouts)"), ("C_test", "C (stochastic)"),
                   ("D_test", "D (long corridors)")]:
        if f"ood__{k}__E" in d:
            E_by[lab] = d[f"ood__{k}__E"]
    F.fig2_error_vs_horizon(E_by, out)

    # Fig 3: Claim A -- the strongest single uncertainty scalar vs future error
    best_sig = ana.get("best_ceiling", "imagined_disp_spread")
    key = f"sig__{best_sig}"
    if key in d:
        F.fig3_uncertainty_vs_error(d[key], d["E_test"], best_sig, out)

    # Fig 4: THE headline figure
    extra = {}
    ceil_key = f"ceil__{best_sig}"
    if ceil_key in d:
        extra[f"Claim-A ceiling\n({best_sig})"] = d[ceil_key]
    if "pred__B8b_privileged_state" in d:
        extra["privileged state"] = d["pred__B8b_privileged_state"]
    F.fig4_predicted_vs_actual(pred, y, out, extra,
                               main_label=f"$g_\\phi$ ({primary.replace('_', ' ')})")

    # Fig 5: calibration
    pp = res / f"probs_{primary}_seed{s}.npy"
    cal = F.fig5_calibration(np.load(pp), y, out) if pp.exists() else None

    # Fig 7: OOD
    rec = {}
    for k, lab in [("B_test", "B unseen"), ("C_test", "C stochastic"), ("D_test", "D corridors")]:
        p = res / f"ood_pred_{k}_seed{s}.npy"
        if p.exists() and f"ood__{k}__y" in d:
            rec[lab] = {"y": d[f"ood__{k}__y"], "pred": np.load(p)}
    rec = {"A in-dist": {"y": y, "pred": pred}, **rec}
    F.fig7_ood(rec, out)

    # Fig 8: failure analysis
    fail = F.fig8_failures(pred, y, d["E_test"], out)

    # Fig 6: planning (only if the planning experiment has been run)
    pl = res / "planning.json"
    if pl.exists():
        F.fig6_planning(json.load(open(pl)), out)

    summary = {"primary_variant": primary, "failure_stats": fail,
               "calibration_ece": {str(k): v["ece"] for k, v in cal.items()} if cal else None}
    json.dump(summary, open(out / "figure_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("figures written to", out)
    for f in sorted(out.glob("*.png")):
        print("  ", f.relative_to(ROOT))


if __name__ == "__main__":
    main()
