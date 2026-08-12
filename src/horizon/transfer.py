"""H6: transfer matrix for simulation metacognition.

Frozen in PREREGISTRATION_H6.md. No world-model retraining: only g_phi is refitted over
features already computed in the main run.

Contamination control: each OOD config's episodes are partitioned deterministically by
episode id (60/20/20). The 20% test portion is evaluation-only in EVERY regime, so a config
can appear in training without ever leaking into its own evaluation.
"""
from __future__ import annotations

import numpy as np
import torch

from src.evaluation.analysis import VARIANTS, cols_for
from src.evaluation.stats import calibration_curve, full_metrics, paired_diff_ci, mae, spearman
from src.horizon.predictor import train_predictor
from src.uncertainty.signals import SCALAR_SIGNALS

BASE_VARIANT = "no_latent"          # frozen
CONFIGS = ["A", "B", "C", "D"]
REGIMES = {                          # frozen
    "A": ["A"],
    "A+B": ["A", "B"],
    "A+C": ["A", "C"],
    "A+D": ["A", "D"],
    "A+B+C": ["A", "B", "C"],
}
CAL_HORIZONS = [3, 5, 10, 20]


def partition_ood(groups: np.ndarray) -> dict[str, np.ndarray]:
    """Deterministic 60/20/20 split of an OOD config, by episode id."""
    m = groups % 5
    return {"train": np.isin(m, [0, 1, 2]), "val": m == 3, "test": m == 4}


def build_pools(S: dict, base_cols: np.ndarray) -> dict:
    """Per-config train/val/test pools of (X, y, groups), with globally unique group ids."""
    pools: dict[str, dict[str, dict]] = {}
    for ci, cfg in enumerate(CONFIGS):
        offset = (ci + 1) * 10_000_000     # keep episode ids unique across configs
        if cfg == "A":
            spec = {"train": "A_gtrain", "val": "A_val", "test": "A_test"}
            pools["A"] = {
                part: {"X": S[key]["X"][:, base_cols], "y": S[key]["y"],
                       "groups": S[key]["groups"] + offset}
                for part, key in spec.items()
            }
        else:
            key = f"{cfg}_test"
            masks = partition_ood(S[key]["groups"])
            pools[cfg] = {
                part: {"X": S[key]["X"][m][:, base_cols], "y": S[key]["y"][m],
                       "groups": S[key]["groups"][m] + offset}
                for part, m in masks.items()
            }
    return pools


def _concat(pools: dict, cfgs: list[str], part: str) -> dict:
    return {k: np.concatenate([pools[c][part][k] for c in cfgs]) for k in ("X", "y", "groups")}


def run_transfer(S: dict, names: list[str], groups_map: dict, seed: int, cfg_obj,
                 device: str = "cpu", n_boot: int = 2000, size_matched: bool = False) -> dict:
    nF = S["A_test"]["X"].shape[1]
    base_cols = cols_for(groups_map, VARIANTS[BASE_VARIANT], nF)
    pools = build_pools(S, base_cols)

    n_ref = len(pools["A"]["train"]["y"])   # for the size-matched control
    rng = np.random.default_rng(seed)

    out: dict = {"seed": seed, "size_matched": size_matched,
                 "n_train_per_regime": {}, "n_test_per_config":
                     {c: int(len(pools[c]["test"]["y"])) for c in CONFIGS}}

    preds: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, dict]] = {}

    for rname, rcfgs in REGIMES.items():
        tr = _concat(pools, rcfgs, "train")
        va = _concat(pools, rcfgs, "val")
        if size_matched and len(tr["y"]) > n_ref:
            sel = rng.choice(len(tr["y"]), n_ref, replace=False)
            tr = {k: v[sel] for k, v in tr.items()}
        out["n_train_per_regime"][rname] = int(len(tr["y"]))

        model, std, _ = train_predictor(
            tr["X"], tr["y"], va["X"], va["y"], seed=seed, epochs=cfg_obj.g_epochs,
            hidden=cfg_obj.g_hidden, depth=cfg_obj.g_depth, lr=cfg_obj.g_lr, device=device)

        preds[rname], metrics[rname] = {}, {}
        for c in CONFIGS:
            te = pools[c]["test"]
            x = torch.as_tensor(std(te["X"]), device=device)
            p, prob = model.predict(x)
            preds[rname][c] = p
            met = full_metrics(p, te["y"], te["groups"], n_boot=n_boot, seed=seed)
            cal = calibration_curve(prob, te["y"], CAL_HORIZONS)
            metrics[rname][c] = {
                "pred_mean": float(p.mean()), "actual_mean": float(te["y"].mean()),
                "mae": met["mae"], "spearman": met["spearman"], "r2": met["r2"],
                "ece": {str(h): cal[h]["ece"] for h in CAL_HORIZONS},
                "seen_in_training": c in rcfgs,
            }
    out["metrics"] = metrics

    # ---- transfer matrix vs the IID-only baseline ----
    transfer = {}
    for rname in REGIMES:
        if rname == "A":
            continue
        transfer[rname] = {}
        for c in CONFIGS:
            te = pools[c]["test"]
            d = paired_diff_ci(mae, te["groups"], preds[rname][c], preds["A"][c], te["y"],
                               n_boot=n_boot, seed=seed)
            transfer[rname][c] = {
                "T_MAE": d,                       # positive => regime beats g_A on config c
                "T_rho": (metrics[rname][c]["spearman"]["point"]
                          - metrics["A"][c]["spearman"]["point"]),
                "seen_in_training": c in REGIMES[rname],
            }
    out["transfer"] = transfer

    # ---- rank stability of each uncertainty feature across distributions ----
    rank = {}
    feats = dict(SCALAR_SIGNALS)
    feats["latent_novelty"] = "latent_nn_dist"
    for label, fname in feats.items():
        if fname not in names:
            continue
        j = names.index(fname)
        rhos = {}
        for c in CONFIGS:
            key = "A_test" if c == "A" else f"{c}_test"
            if c == "A":
                x, y = S[key]["X"][:, j], S[key]["y"]
            else:
                m = partition_ood(S[key]["groups"])["test"]
                x, y = S[key]["X"][m, j], S[key]["y"][m]
            rhos[c] = spearman(x, y)
        vals = np.array(list(rhos.values()))
        rank[label] = {"rho": rhos, "rank_stability_var": float(vals.var()),
                       "abs_rho_A": float(abs(rhos["A"])),
                       "mean_abs_rho_ood": float(np.mean([abs(rhos[c]) for c in "BCD"]))}
    out["rank_stability"] = rank
    return out


def evaluate_criteria(res: dict, tol_id: float = 0.5) -> dict:
    """Frozen H6 criteria (PREREGISTRATION_H6.md)."""
    M, T = res["metrics"], res["transfer"]
    a_mae = M["A"]["A"]["mae"]["point"]
    cells = {}
    for rname in T:
        for c in CONFIGS:
            if c in REGIMES[rname]:
                continue                      # only completely held-out configs count
            c1 = M[rname][c]["pred_mean"] < M[rname]["A"]["pred_mean"]
            c2 = M[rname][c]["spearman"]["point"] > 0.3
            c3 = T[rname][c]["T_MAE"]["lo"] > 0
            cells[f"{rname}->{c}"] = {
                "c1_more_conservative_than_ID": bool(c1),
                "c2_spearman_gt_0.3": bool(c2),
                "c3_beats_gA_MAE": bool(c3),
                "restored": bool(c1 and c2 and c3),
            }
    id_ok = {r: bool(M[r]["A"]["mae"]["point"] - a_mae <= tol_id) for r in M}
    critical = cells.get("A+B+C->D", {}).get("restored", False)
    any_restored = [k for k, v in cells.items() if v["restored"]]
    verdict = ("SUPPORTED" if critical else
               "PARTIALLY SUPPORTED" if any_restored else "NOT SUPPORTED")
    return {"cells": cells, "id_preserved": id_ok, "critical_cell_restored": bool(critical),
            "restored_cells": any_restored, "verdict": verdict}
