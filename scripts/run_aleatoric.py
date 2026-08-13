"""H9: does the aleatoric family transfer, using instruments that actually bite?

Frozen in PREREGISTRATION_H9.md. Two stages:
  1. CALIBRATION -- measure H* for a ladder of severities on disjoint layout seeds, pick the
     rate whose H* is closest to config B's 7.75 (a target fixed by an existing measurement).
  2. TEST -- collect the selected configs on the standard OOD seeds, apply the validity gate,
     then score the frozen predictions Q1-Q6.

No world-model retraining.
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
from src.envs.configs import aleatoric_config  # noqa: E402
from src.evaluation.analysis import VARIANTS, cols_for, degeneracy_report  # noqa: E402
from src.evaluation.stats import full_metrics, spearman  # noqa: E402
from src.horizon.predictor import train_predictor  # noqa: E402
from src.horizon.transfer import build_pools, partition_ood  # noqa: E402
from src.models.train import load_member  # noqa: E402
from src.utils.seeding import get_device  # noqa: E402
from scripts.run_experiment import extract_split  # noqa: E402

TARGET_HSTAR = 7.75           # config B's H*, fixed in advance
LADDER = {"obs": [0.30, 0.50, 0.70], "dropout": [0.20, 0.35, 0.50]}
CAL_SEEDS = (200000, 200200)  # disjoint from every scored test set
NEW = ["F2", "F3"]
REGIMES = {"A": ["A"], "A+B": ["A", "B"], "A+C": ["A", "C"],
           "A+F2": ["A", "F2"], "A+F3": ["A", "F3"]}
PREDICTIONS = [
    ("Q1", "A+C", "F2", "gt", 0.50), ("Q2", "A+B", "F2", "lt", 0.35),
    ("Q3", "A+F2", "C", "gt", 0.50), ("Q4", "A+C", "F3", "gt", 0.50),
    ("Q5", "A+B", "F3", "lt", 0.35),
]
# validity gate (frozen)
GATE_HSTAR_MAX = 9.3          # 40% below A's 15.52
GATE_BASELINE_RHO_MAX = 0.35


def _extract(models, traj, cfg, device, tag):
    return extract_split(models, traj, cfg, device, tag)


def calibrate(models0, cfg, device, out: Path) -> dict:
    """Stage 1: pick each mechanism's severity by the frozen rule."""
    cfg_cal = EXP_CONFIGS[cfg.name]
    cal = dict(cfg_cal.__dict__)
    chosen, table = {}, []
    for kind, rates in LADDER.items():
        best = None
        for p in rates:
            ec = aleatoric_config(f"CAL_{kind}_{p}", kind, p)
            tr = collect(ec, list(range(*CAL_SEEDS)),
                         steps_per_episode=cfg.steps_per_episode,
                         epsilon=cfg.policy_epsilon, policy_seed=CAL_SEEDS[0])
            sp = _extract(models0, tr, cfg, device, f"cal:{kind}:{p}")
            h = float(sp["y"].mean())
            table.append({"kind": kind, "p": p, "H_star": h,
                          "dist_to_target": abs(h - TARGET_HSTAR)})
            print(f"  calibration {kind} p={p}: H* {h:.2f} "
                  f"(target {TARGET_HSTAR}, |diff| {abs(h-TARGET_HSTAR):.2f})", flush=True)
            if best is None or abs(h - TARGET_HSTAR) < best[1]:
                best = (p, abs(h - TARGET_HSTAR), h)
        chosen[kind] = {"p": best[0], "H_star_cal": best[2]}
        print(f"  -> selected {kind} p={best[0]} (H* {best[2]:.2f})", flush=True)
    json.dump({"ladder": table, "chosen": chosen, "target": TARGET_HSTAR},
              open(out / "calibration.json", "w"), indent=2)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--results", default=None)
    ap.add_argument("--cal-anchors", type=int, default=3000)
    args = ap.parse_args()
    cfg = EXP_CONFIGS[args.config]
    device = get_device()
    res = Path(args.results) if args.results else ROOT / "results" / cfg.name
    out = ROOT / "results" / "aleatoric"
    out.mkdir(parents=True, exist_ok=True)
    ck = ROOT / "checkpoints" / cfg.name
    print("device:", device, flush=True)

    models0 = [load_member(ck / f"wm_seed0_member{k}.pt", device)
               for k in range(cfg.ensemble_k)]

    # ---------------- stage 1: calibration ----------------
    print("\n=== stage 1: severity calibration (seed 0, disjoint layout seeds) ===", flush=True)
    cal_cfg = EXP_CONFIGS[args.config]
    saved_cap, saved_stride = cal_cfg.max_anchors_per_split, cal_cfg.anchor_stride
    cal_cfg.max_anchors_per_split, cal_cfg.anchor_stride = args.cal_anchors, 16
    chosen = calibrate(models0, cal_cfg, device, out)
    cal_cfg.max_anchors_per_split, cal_cfg.anchor_stride = saved_cap, saved_stride

    selected = {"F2": aleatoric_config("F2", "obs", chosen["obs"]["p"]),
                "F3": aleatoric_config("F3", "dropout", chosen["dropout"]["p"])}

    # ---------------- stage 2: test ----------------
    print("\n=== stage 2: scored test sets ===", flush=True)
    lo, hi = cfg.ood_seeds
    new_traj = {}
    for n in NEW:
        p = ROOT / "experiments" / "aleatoric" / f"traj_{n}.npz"
        if p.exists():
            new_traj[n] = Trajectories.load(p)
        else:
            tr = collect(selected[n], list(range(lo, hi)),
                         steps_per_episode=cfg.steps_per_episode,
                         epsilon=cfg.policy_epsilon, policy_seed=lo)
            tr.save(p)
            new_traj[n] = tr
        print(f"  {n} ({selected[n].description}): {len(new_traj[n])} episodes", flush=True)

    per_seed = []
    for seed in args.seeds:
        models = [load_member(ck / f"wm_seed{seed}_member{k}.pt", device)
                  for k in range(cfg.ensemble_k)]
        base = np.load(res / f"features_seed{seed}.npz", allow_pickle=True)
        names = [str(s) for s in base["names"]]
        groups_map = {k: list(map(int, v)) for k, v in
                      json.load(open(res / f"feature_groups_seed{seed}.json")).items()}
        S = {}
        for key in ["A_gtrain", "A_val", "A_test", "B_test", "C_test", "D_test"]:
            S[key] = {f: base[f"{key}__{f}"] for f in
                      ["X", "y", "y_nonmono", "y_m0", "E", "groups", "starts", "priv", "acts"]}

        # append latent_nn_dist so new configs match the stored 634-column layout
        ref_cols = [i for i, n in enumerate(names) if n.startswith("h0_")]
        R = S["A_gtrain"]["X"][:, ref_cols]
        rng_ref = np.random.default_rng(0)
        if len(R) > 4000:
            R = R[rng_ref.choice(len(R), 4000, replace=False)]
        Rt = torch.as_tensor(R)
        for n in NEW:
            sp = _extract(models, new_traj[n], cfg, device, f"{n}_test")
            q = [i for i, nm in enumerate(sp["names"]) if nm.startswith("h0_")]
            d = torch.cdist(torch.as_tensor(sp["X"][:, q]), Rt).min(1).values.numpy()
            sp["X"] = np.concatenate([sp["X"], d[:, None].astype(np.float32)], 1)
            sp["names"] = list(sp["names"]) + ["latent_nn_dist"]
            S[f"{n}_test"] = sp
            assert sp["X"].shape[1] == S["A_test"]["X"].shape[1]

        nF = S["A_test"]["X"].shape[1]
        base_cols = cols_for(groups_map, VARIANTS["no_latent"], nF)
        pools = build_pools(S, base_cols)
        for n in NEW:
            key = f"{n}_test"
            masks = partition_ood(S[key]["groups"])
            off = (ord(n[-1]) + 90) * 10_000_000
            pools[n] = {part: {"X": S[key]["X"][m][:, base_cols], "y": S[key]["y"][m],
                               "groups": S[key]["groups"][m] + off}
                        for part, m in masks.items()}

        rec = {"seed": seed, "rho": {}, "degeneracy": {}}
        for n in NEW:
            rec["degeneracy"][n] = degeneracy_report(S[f"{n}_test"]["y"])
        for rname, rcfgs in REGIMES.items():
            tr_ = {k: np.concatenate([pools[c]["train"][k] for c in rcfgs])
                   for k in ("X", "y", "groups")}
            va_ = {k: np.concatenate([pools[c]["val"][k] for c in rcfgs])
                   for k in ("X", "y", "groups")}
            m, std, _ = train_predictor(tr_["X"], tr_["y"], va_["X"], va_["y"], seed=seed,
                                        epochs=cfg.g_epochs, hidden=cfg.g_hidden,
                                        depth=cfg.g_depth, lr=cfg.g_lr, device=str(device))
            for c in ["A", "B", "C", "D", "F2", "F3"]:
                te = pools[c]["test"]
                p_ = m.predict(torch.as_tensor(std(te["X"]), device=device))[0]
                met = full_metrics(p_, te["y"], te["groups"], n_boot=cfg.n_boot, seed=seed)
                rec["rho"][f"{rname}->{c}"] = met["spearman"]["point"]
        j = names.index("free_ahead")
        rec["free_ahead_rho"] = {}
        for c in NEW:
            m_ = partition_ood(S[f"{c}_test"]["groups"])["test"]
            rec["free_ahead_rho"][c] = spearman(S[f"{c}_test"]["X"][m_, j],
                                                S[f"{c}_test"]["y"][m_])
        per_seed.append(rec)
        json.dump(rec, open(out / f"aleatoric_seed{seed}.json", "w"), indent=2, default=float)

    # ---------------- report ----------------
    print("\n" + "=" * 72 + "\n H9 aleatoric-family transfer\n" + "=" * 72)
    print(f"\nselected severities: F2 obs p={chosen['obs']['p']}, "
          f"F3 dropout p={chosen['dropout']['p']}\n")

    print("### Validity gate (frozen: H* <= 9.3 AND IID-only transfer < 0.35)\n")
    print("| config | H* | A->cfg rho | passes gate |")
    print("|---|---|---|---|")
    valid = {}
    for n in NEW:
        h = float(np.mean([r["degeneracy"][n]["mean"] for r in per_seed]))
        b = float(np.mean([r["rho"][f"A->{n}"] for r in per_seed]))
        ok = (h <= GATE_HSTAR_MAX) and (b < GATE_BASELINE_RHO_MAX)
        valid[n] = ok
        print(f"| {n} | {h:.2f} | {b:+.3f} | {'YES' if ok else 'NO (invalid instrument)'} |")

    cols = ["A", "B", "C", "D", "F2", "F3"]
    print("\n### Spearman by regime (mean over seeds)\n")
    print("| regime | " + " | ".join(f"test {c}" for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for rname in REGIMES:
        cells = []
        for c in cols:
            v = float(np.mean([r["rho"][f"{rname}->{c}"] for r in per_seed]))
            s = f"{v:+.3f}"
            cells.append(s if c in REGIMES[rname] else f"**{s}**")
        print(f"| {rname} | " + " | ".join(cells) + " |")

    print("\n### Frozen predictions\n")
    print("| # | cell | threshold | rho | seeds | holds | instrument |")
    print("|---|---|---|---|---|---|---|")
    bits = {}
    for pid, reg, tgt, direction, thr in PREDICTIONS:
        vals = [r["rho"][f"{reg}->{tgt}"] for r in per_seed]
        passes = [(v > thr) if direction == "gt" else (v < thr) for v in vals]
        holds = sum(passes) >= 2
        bits[pid] = holds and valid.get(tgt, valid.get(reg.split("+")[-1], True))
        arrow = ">" if direction == "gt" else "<"
        inst = "valid" if valid.get(tgt, True) else "INVALID"
        print(f"| {pid} | {reg} -> {tgt} | rho {arrow} {thr:.2f} | {np.mean(vals):+.3f} | "
              f"{sum(passes)}/{len(passes)} | {'YES' if holds else 'no'} | {inst} |")

    fr = {c: float(np.mean([r["free_ahead_rho"][c] for r in per_seed])) for c in NEW}
    q6 = all(abs(v) < 0.15 for v in fr.values())
    print(f"\n### Q6 `free_ahead` on new configs: " +
          ", ".join(f"{c} {v:+.3f}" for c, v in fr.items()) + f"  -> holds: {q6}")

    if not any(valid.values()):
        verdict = "INCONCLUSIVE (no valid instrument)"
    elif bits.get("Q1") and bits.get("Q2"):
        verdict = "SUPPORTED (aleatoric family real and distinct)"
    elif bits.get("Q1"):
        verdict = "PARTIALLY SUPPORTED (transfer real, but not family-specific -> severity account)"
    else:
        verdict = "NOT SUPPORTED (aleatoric shifts do not transfer to each other)"
    print(f"\n### H9 verdict: **{verdict}**")
    json.dump({"verdict": verdict, "valid": valid, "predictions": bits,
               "free_ahead": fr, "chosen": chosen},
              open(out / "aleatoric_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
