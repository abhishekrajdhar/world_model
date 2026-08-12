"""H8: does the epistemic/aleatoric taxonomy PREDICT transfer to new configs E and F?

Predictions are frozen in PREREGISTRATION_H8.md. Uses the main-run world models unchanged
(egocentric observations transfer to any layout by construction), so only data collection and
feature extraction are new.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.experiment import CONFIGS as EXP_CONFIGS  # noqa: E402
from src.data.collect import Trajectories, collect  # noqa: E402
from src.envs.configs import ENV_CONFIGS  # noqa: E402
from src.evaluation.analysis import VARIANTS, cols_for, degeneracy_report  # noqa: E402
from src.evaluation.stats import full_metrics, spearman  # noqa: E402
from src.horizon.predictor import train_predictor  # noqa: E402
from src.horizon.transfer import build_pools, partition_ood  # noqa: E402
from src.models.train import load_member  # noqa: E402
from src.utils.seeding import get_device  # noqa: E402
from scripts.run_experiment import add_latent_nn_distance, extract_split  # noqa: E402

NEW = ["E", "F"]
# regimes needed to score P1..P5
REGIMES = {
    "A": ["A"], "A+B": ["A", "B"], "A+C": ["A", "C"],
    "A+E": ["A", "E"], "A+F": ["A", "F"],
}
PREDICTIONS = [
    ("P1", "A+B", "E", "gt", 0.50),
    ("P2", "A+B", "F", "lt", 0.35),
    ("P3", "A+C", "F", "gt", 0.50),
    ("P4", "A+E", "D", "gt", 0.50),
    ("P5", "A+F", "C", "gt", 0.50),
]


def collect_new(cfg, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    got = {}
    for name in NEW:
        p = out_dir / f"traj_{name}_test.npz"
        if p.exists():
            got[name] = Trajectories.load(p)
        else:
            lo, hi = cfg.ood_seeds
            print(f"collecting {name} ({ENV_CONFIGS[name].env_id}, "
                  f"noise={ENV_CONFIGS[name].obs_noise}, slip={ENV_CONFIGS[name].slip})",
                  flush=True)
            tr = collect(ENV_CONFIGS[name], list(range(lo, hi)),
                         steps_per_episode=cfg.steps_per_episode,
                         epsilon=cfg.policy_epsilon, policy_seed=lo)
            tr.save(p)
            got[name] = tr
        print(f"  {name}: {len(got[name])} episodes, {len(got[name].image)} steps", flush=True)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    cfg = EXP_CONFIGS[args.config]
    device = get_device()
    res = Path(args.results) if args.results else ROOT / "results" / cfg.name
    out = ROOT / "results" / "taxonomy"
    out.mkdir(parents=True, exist_ok=True)
    print("device:", device, flush=True)

    new_traj = collect_new(cfg, ROOT / "experiments" / "taxonomy")

    per_seed = []
    for seed in args.seeds:
        ck = ROOT / "checkpoints" / cfg.name
        models = [load_member(ck / f"wm_seed{seed}_member{k}.pt", device)
                  for k in range(cfg.ensemble_k)]
        print(f"--- seed {seed}: extracting E/F features ---", flush=True)
        base = np.load(res / f"features_seed{seed}.npz", allow_pickle=True)
        names = [str(s) for s in base["names"]]
        groups_map = {k: list(map(int, v)) for k, v in
                      json.load(open(res / f"feature_groups_seed{seed}.json")).items()}

        S = {}
        for key in ["A_gtrain", "A_val", "A_test", "B_test", "C_test", "D_test"]:
            S[key] = {f: base[f"{key}__{f}"] for f in
                      ["X", "y", "y_nonmono", "y_m0", "E", "groups", "starts", "priv", "acts"]}
        newsplits = {}
        for name in NEW:
            sp = extract_split(models, new_traj[name], cfg, device, f"{name}_test")
            newsplits[f"{name}_test"] = sp
        # append the latent-novelty column consistently with the stored features
        add_latent_nn_distance({**{k: dict(v) for k, v in newsplits.items()},
                                "A_gtrain": {"X": S["A_gtrain"]["X"], "names": names,
                                             "feature_groups": groups_map}},
                               ref_key="A_gtrain")
        for k, v in newsplits.items():
            S[k] = v
        assert S["E_test"]["X"].shape[1] == S["A_test"]["X"].shape[1], (
            S["E_test"]["X"].shape, S["A_test"]["X"].shape)

        rec = {"seed": seed,
               "degeneracy": {n: degeneracy_report(S[f"{n}_test"]["y"]) for n in NEW}}
        nF = S["A_test"]["X"].shape[1]
        base_cols = cols_for(groups_map, VARIANTS["no_latent"], nF)
        pools = build_pools(S, base_cols)
        # E and F need pools too (build_pools only knows A-D)
        for n in NEW:
            key = f"{n}_test"
            masks = partition_ood(S[key]["groups"])
            off = (ord(n) + 50) * 10_000_000
            pools[n] = {part: {"X": S[key]["X"][m][:, base_cols], "y": S[key]["y"][m],
                               "groups": S[key]["groups"][m] + off}
                        for part, m in masks.items()}

        rec["rho"] = {}
        rec["metrics"] = {}
        for rname, rcfgs in REGIMES.items():
            tr = {k: np.concatenate([pools[c]["train"][k] for c in rcfgs])
                  for k in ("X", "y", "groups")}
            va = {k: np.concatenate([pools[c]["val"][k] for c in rcfgs])
                  for k in ("X", "y", "groups")}
            m, std, _ = train_predictor(tr["X"], tr["y"], va["X"], va["y"], seed=seed,
                                        epochs=cfg.g_epochs, hidden=cfg.g_hidden,
                                        depth=cfg.g_depth, lr=cfg.g_lr, device=str(device))
            for c in ["A", "B", "C", "D", "E", "F"]:
                te = pools[c]["test"]
                p = m.predict(torch.as_tensor(std(te["X"]), device=device))[0]
                met = full_metrics(p, te["y"], te["groups"], n_boot=cfg.n_boot, seed=seed)
                rec["rho"][f"{rname}->{c}"] = met["spearman"]["point"]
                rec["metrics"][f"{rname}->{c}"] = {
                    "mae": met["mae"]["point"], "rho": met["spearman"]["point"],
                    "pred_mean": float(p.mean()), "actual_mean": float(te["y"].mean()),
                    "seen": c in rcfgs}

        # P6 diagnostic: is free_ahead informative on E and not on F?
        j = names.index("free_ahead")
        rec["free_ahead_rho"] = {}
        for c in ["A", "B", "C", "D", "E", "F"]:
            key = "A_test" if c == "A" else f"{c}_test"
            if c == "A":
                x, y = S[key]["X"][:, j], S[key]["y"]
            else:
                m_ = partition_ood(S[key]["groups"])["test"]
                x, y = S[key]["X"][m_, j], S[key]["y"][m_]
            rec["free_ahead_rho"][c] = spearman(x, y)
        per_seed.append(rec)
        json.dump(rec, open(out / f"taxonomy_seed{seed}.json", "w"), indent=2, default=float)

    # ---------------- report ----------------
    print("\n" + "=" * 72 + "\n H8 taxonomy test\n" + "=" * 72)
    print("\n### Degeneracy check on the new configs\n")
    print("| config | mean H* | sd | IQR | ceiling | zero | degenerate |")
    print("|---|---|---|---|---|---|---|")
    for n in NEW:
        d = per_seed[0]["degeneracy"][n]
        print(f"| {n} | {d['mean']:.2f} | {d['sd']:.2f} | {d['iqr']:.1f} | "
              f"{d['ceiling_frac']:.3f} | {d['zero_frac']:.3f} | {d['degenerate']} |")

    print("\n### Spearman by regime (mean over seeds)\n")
    cols = ["A", "B", "C", "D", "E", "F"]
    print("| regime | " + " | ".join(f"test {c}" for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for rname in REGIMES:
        cells = []
        for c in cols:
            v = float(np.mean([r["rho"][f"{rname}->{c}"] for r in per_seed]))
            s = f"{v:+.3f}"
            cells.append(s if c in REGIMES[rname] else f"**{s}**")
        print(f"| {rname} | " + " | ".join(cells) + " |")

    print("\n### Registered predictions\n")
    print("| # | cell | threshold | rho (mean) | per-seed pass | holds |")
    print("|---|---|---|---|---|---|")
    verdict_bits = {}
    for pid, reg, tgt, direction, thr in PREDICTIONS:
        vals = [r["rho"][f"{reg}->{tgt}"] for r in per_seed]
        passes = [(v > thr) if direction == "gt" else (v < thr) for v in vals]
        holds = sum(passes) >= 2
        verdict_bits[pid] = holds
        arrow = ">" if direction == "gt" else "<"
        print(f"| {pid} | {reg} -> {tgt} | rho {arrow} {thr:.2f} | {np.mean(vals):+.3f} | "
              f"{sum(passes)}/{len(passes)} | {'YES' if holds else 'NO'} |")

    print("\n### P6: is `free_ahead` the epistemic carrier?\n")
    fr = {c: float(np.mean([r["free_ahead_rho"][c] for r in per_seed]))
          for c in cols}
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    print("| " + " | ".join(f"{fr[c]:+.3f}" for c in cols) + " |")
    p6 = abs(fr["E"]) > 0.3 and abs(fr["F"]) < 0.15
    print(f"\nP6 holds: {p6}  (|rho_E|={abs(fr['E']):.3f} > 0.3, "
          f"|rho_F|={abs(fr['F']):.3f} < 0.15)")

    if any(per_seed[0]["degeneracy"][n]["degenerate"] for n in NEW):
        verdict = "INCONCLUSIVE (degenerate H* on a new config)"
    elif verdict_bits["P1"] and verdict_bits["P2"] and verdict_bits["P3"]:
        verdict = "SUPPORTED"
    elif verdict_bits["P1"] and verdict_bits["P2"]:
        verdict = "PARTIALLY SUPPORTED (taxonomy too coarse; P3 failed)"
    else:
        verdict = "NOT SUPPORTED"
    print(f"\n### H8 verdict: **{verdict}**")
    json.dump({"verdict": verdict, "predictions": verdict_bits, "P6": bool(p6),
               "free_ahead_rho": fr},
              open(out / "taxonomy_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
