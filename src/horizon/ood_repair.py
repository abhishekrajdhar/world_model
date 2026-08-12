"""H5: can the OOD failure of the horizon predictor be repaired without refitting on OOD?

Arms are frozen in PREREGISTRATION_H5.md:
    R0 baseline | R1 rank-normalised | R2 novelty-gated | R3 combined

Nothing here is fitted on B/C/D. The gate constants come from A-val by a fixed rule.
"""
from __future__ import annotations

import numpy as np
import torch

from src.evaluation.analysis import VARIANTS, cols_for
from src.evaluation.stats import full_metrics, spearman
from src.horizon.predictor import train_predictor

BASE_VARIANT = "no_latent"          # frozen: best variant in the main run
UNC_GROUPS = ["apriori", "imag"]    # features that get rank-normalised in R1/R3
OOD_SPLITS = ["B_test", "C_test", "D_test"]


def rank_normalize(X: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Replace each selected column by its within-array quantile rank in [0, 1].

    Transductive by construction (see PREREGISTRATION_H5.md, 'Honest limitation').
    """
    out = X.copy()
    n = len(X)
    if n < 2:
        return out
    for j in cols:
        order = np.argsort(X[:, j], kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n)
        out[:, j] = (ranks / (n - 1)).astype(X.dtype)
    return out


def gate_params(nu_val: np.ndarray) -> tuple[float, float]:
    """tau = 95th pct of novelty on A-val; lambda fixed so the 99th pct shrinks by 10%."""
    q95, q99 = np.percentile(nu_val, [95, 99])
    span = max(float(q99 - q95), 1e-8)
    lam = -np.log(0.9) / span
    return float(q95), float(lam)


def apply_gate(pred: np.ndarray, nu: np.ndarray, tau: float, lam: float) -> np.ndarray:
    return pred * np.exp(-lam * np.maximum(0.0, nu - tau))


def _novelty_col(names: list[str]) -> int:
    if "latent_nn_dist" not in names:
        raise KeyError("latent_nn_dist feature missing; cannot run the novelty gate")
    return names.index("latent_nn_dist")


def run_arms(S: dict, names: list[str], groups_map: dict, seed: int, cfg, device="cpu",
             n_boot: int = 2000) -> dict:
    """Train all four arms and evaluate on A-test + the three OOD splits."""
    nF = S["A_test"]["X"].shape[1]
    base_cols = cols_for(groups_map, VARIANTS[BASE_VARIANT], nF)
    unc_cols_global = cols_for(groups_map, UNC_GROUPS, nF)
    # positions of the uncertainty columns *within* the base feature subset
    pos = {c: i for i, c in enumerate(base_cols)}
    unc_local = np.array([pos[c] for c in unc_cols_global if c in pos], dtype=np.int64)
    nu_col = _novelty_col(names)

    def sub(split, rank=False):
        X = S[split]["X"][:, base_cols]
        return rank_normalize(X, unc_local) if rank else X

    results: dict = {"seed": seed, "n_base_features": int(len(base_cols)),
                     "n_rank_normalised": int(len(unc_local))}

    # --- diagnostic: do the uncertainty signals mislead, or merely rescale? ---
    diag = {}
    for feat in ["dispspread_mean", "js_moved_mean", "ent_moved_mean", "prior_std_anchor",
                 "trunc_h_disp", "latent_nn_dist"]:
        if feat not in names:
            continue
        j = names.index(feat)
        diag[feat] = {sp: {"mean": float(S[sp]["X"][:, j].mean()),
                           "sd": float(S[sp]["X"][:, j].std()),
                           "rho_with_Hstar_within_split": spearman(S[sp]["X"][:, j], S[sp]["y"])}
                      for sp in ["A_test", *OOD_SPLITS]}
    results["diagnostic"] = diag

    # --- gate constants from A-val only ---
    tau, lam = gate_params(S["A_val"]["X"][:, nu_col])
    results["gate"] = {"tau": tau, "lambda": lam}

    # --- train the two predictors (plain features, rank-normalised features) ---
    models = {}
    for rank in (False, True):
        m, std, hist = train_predictor(
            sub("A_gtrain", rank), S["A_gtrain"]["y"], sub("A_val", rank), S["A_val"]["y"],
            seed=seed, epochs=cfg.g_epochs, hidden=cfg.g_hidden, depth=cfg.g_depth,
            lr=cfg.g_lr, device=device)
        models[rank] = (m, std)
        results[f"val_mae_rank{int(rank)}"] = float(np.min(hist["val_mae"]))

    def predict(split, rank):
        m, std = models[rank]
        x = torch.as_tensor(std(sub(split, rank)), device=device)
        return m.predict(x)[0]

    arms = {}
    for arm, rank, gated in [("R0_baseline", False, False), ("R1_ranknorm", True, False),
                             ("R2_gated", False, True), ("R3_combined", True, True)]:
        rec = {}
        for split in ["A_test", *OOD_SPLITS]:
            p = predict(split, rank)
            if gated:
                p = apply_gate(p, S[split]["X"][:, nu_col], tau, lam)
            met = full_metrics(p, S[split]["y"], S[split]["groups"], n_boot=n_boot, seed=seed)
            rec[split] = {
                "pred_mean": float(p.mean()),
                "actual_mean": float(S[split]["y"].mean()),
                "mae": met["mae"], "spearman": met["spearman"], "r2": met["r2"],
            }
        arms[arm] = rec
    results["arms"] = arms
    return results


def evaluate_criteria(arms: dict, tol_mae: float = 0.5) -> dict:
    """Apply the frozen H5 criteria (PREREGISTRATION_H5.md)."""
    r0_a = arms["R0_baseline"]["A_test"]["mae"]["point"]
    out = {}
    for arm, rec in arms.items():
        a_pred = rec["A_test"]["pred_mean"]
        c1 = all(rec[s]["pred_mean"] < a_pred for s in OOD_SPLITS)
        c2 = sum(rec[s]["spearman"]["point"] > 0.3 for s in OOD_SPLITS) >= 2
        c3 = (rec["A_test"]["mae"]["point"] - r0_a) <= tol_mae
        out[arm] = {"c1_horizon_falls_on_all_ood": bool(c1),
                    "c2_spearman_gt_0.3_on_2of3": bool(c2),
                    "c3_id_mae_not_degraded": bool(c3),
                    "supported": bool(c1 and c2 and c3),
                    "partial": bool(c1 and c3 and not c2)}
    repaired = [a for a in ["R1_ranknorm", "R2_gated", "R3_combined"] if out[a]["supported"]]
    partial = [a for a in ["R1_ranknorm", "R2_gated", "R3_combined"] if out[a]["partial"]]
    out["verdict"] = ("SUPPORTED" if repaired else
                      "PARTIALLY SUPPORTED" if partial else "NOT SUPPORTED")
    out["arms_supporting"] = repaired
    out["arms_partial"] = partial
    return out
