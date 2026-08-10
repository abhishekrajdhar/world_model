"""Phases 7-10 + 13: fit the horizon predictor, run all baselines, produce statistics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.envs.configs import N_ACTIONS
from src.evaluation.baselines import build_baselines, raw_signal_direction
from src.evaluation.stats import (bootstrap_ci, full_metrics, mae, paired_diff_ci, spearman)
from src.horizon.predictor import train_predictor
from src.horizon.rih import H_MAX
from src.uncertainty.signals import SCALAR_SIGNALS

# Predictor variants. Each is a subset of the feature groups; comparing them is the ablation
# that says WHICH information carries the horizon signal.
VARIANTS = {
    "full":            ["latent", "apriori", "imag", "actions", "hand"],
    "strict_apriori":  ["latent", "apriori", "actions", "hand"],   # no imagined rollout used
    "no_latent":       ["apriori", "imag", "actions", "hand"],
    "uncertainty_only": ["apriori", "imag"],
    "latent_only":     ["latent"],
    "hand_actions_only": ["actions", "hand"],                       # no world-model internals
    "actions_only":    ["actions"],
}


def load_features(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    names = [str(s) for s in d["names"]]
    splits = {}
    fields = ["X", "y", "y_nonmono", "y_m0", "E", "groups", "starts"]
    for key in ["A_gtrain", "A_val", "A_test", "B_test", "C_test", "D_test"]:
        splits[key] = {f: d[f"{key}__{f}"] for f in fields}
        for opt in ["priv", "acts"]:
            if f"{key}__{opt}" in d:
                splits[key][opt] = d[f"{key}__{opt}"]
    return {"splits": splits, "names": names}


# Object indices that matter for `moved`: passable vs blocking.
_PASSABLE = (1, 3, 8)     # empty, floor, goal
_WALL, _DOOR, _KEY = 2, 4, 5


def privileged_features(sp: dict, action_cols: np.ndarray) -> np.ndarray:
    """PRIVILEGED comparison: the TRUE local grid (including cells the agent cannot see),
    the ORDERED action sequence, and the usual action summaries.

    Note this is an empirical comparison, not a proven ceiling: the predictor still has to
    *learn* when the neural world model errs, which is not a simple function of geometry."""
    p = sp["priv"].reshape(len(sp["priv"]), -1)
    chans = [np.isin(p, _PASSABLE), p == _WALL, p == _DOOR, p == _KEY]
    parts = [np.concatenate(chans, 1).astype(np.float32)]
    if "acts" in sp:  # ordered actions, one-hot: needed to identify the specific path
        a = sp["acts"].astype(np.int64)
        oh = np.zeros((*a.shape, N_ACTIONS), np.float32)
        np.put_along_axis(oh, a[..., None], 1.0, axis=2)
        parts.append(oh.reshape(len(a), -1))
    parts.append(sp["X"][:, action_cols])
    return np.concatenate(parts, 1)


def cols_for(groups_map: dict[str, list[int]], variant: list[str], n_features: int) -> np.ndarray:
    idx = sorted({i for g in variant for i in groups_map.get(g, []) if i < n_features})
    return np.asarray(idx, dtype=np.int64)


def degeneracy_report(y: np.ndarray) -> dict:
    iqr = float(np.percentile(y, 75) - np.percentile(y, 25))
    ceiling = float((y >= H_MAX).mean())
    return {
        "mean": float(y.mean()), "sd": float(y.std()), "iqr": iqr,
        "ceiling_frac": ceiling, "zero_frac": float((y == 0).mean()),
        "degenerate": bool(iqr < 2 or ceiling > 0.80),
    }


def run_analysis(feat_path: Path, groups_map: dict, out_dir: Path, seed: int = 0,
                 device: str = "cpu", cfg=None, n_boot: int = 2000) -> dict:
    data = load_features(feat_path)
    S, names = data["splits"], data["names"]
    nF = S["A_test"]["X"].shape[1]
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"seed": seed, "n_features": nF,
                     "degeneracy": {k: degeneracy_report(S[k]["y"]) for k in S}}

    # ---------- H1 / Claim A: does uncertainty correlate with future error? ----------
    claimA = {}
    col = {n: i for i, n in enumerate(names)}
    for label, feat in SCALAR_SIGNALS.items():
        if feat not in col:
            continue
        u = S["A_test"]["X"][:, col[feat]]
        rec = {"spearman_vs_Hstar": raw_signal_direction(u, S["A_test"]["y"])}
        for h in [1, 5, 10, 20, 30]:
            rec[f"spearman_vs_E{h}"] = raw_signal_direction(u, S["A_test"]["E"][:, h - 1])
        claimA[label] = rec
    results["claim_A"] = claimA

    # ---------- H2: the learned predictor ----------
    y_tr, y_va, y_te = S["A_gtrain"]["y"], S["A_val"]["y"], S["A_test"]["y"]
    g_te = S["A_test"]["groups"]
    preds_test: dict[str, np.ndarray] = {}
    models = {}
    variant_val_mae: dict[str, float] = {}
    for vname, vgroups in VARIANTS.items():
        cols = cols_for(groups_map, vgroups, nF)
        if len(cols) == 0:
            continue
        m, std, hist = train_predictor(
            S["A_gtrain"]["X"][:, cols], y_tr, S["A_val"]["X"][:, cols], y_va,
            seed=seed, epochs=(cfg.g_epochs if cfg else 60),
            hidden=(cfg.g_hidden if cfg else 256), depth=(cfg.g_depth if cfg else 2),
            lr=(cfg.g_lr if cfg else 1e-3), device=device,
        )
        xt = torch.as_tensor(std(S["A_test"]["X"][:, cols]), device=device)
        mean, prob = m.predict(xt)
        preds_test[f"g_{vname}"] = mean
        models[vname] = (m, std, cols)
        variant_val_mae[vname] = float(np.min(hist["val_mae"]))
        np.save(out_dir / f"probs_{vname}_seed{seed}.npy", prob)

    # Privileged upper bound (Baseline 8b): sees the TRUE local grid. Not an agent-available
    # predictor -- it bounds how much of H* is knowable at all from the state + plan.
    if "priv" in S["A_gtrain"]:
        acols = cols_for(groups_map, ["actions"], nF)
        pm_, pstd_, phist = train_predictor(
            privileged_features(S["A_gtrain"], acols), y_tr,
            privileged_features(S["A_val"], acols), y_va,
            seed=seed, epochs=(cfg.g_epochs if cfg else 60), hidden=256, depth=2,
            lr=(cfg.g_lr if cfg else 1e-3), device=device)
        xp = torch.as_tensor(pstd_(privileged_features(S["A_test"], acols)), device=device)
        preds_test["B8b_privileged_state"] = pm_.predict(xp)[0]
        variant_val_mae["privileged_state"] = float(np.min(phist["val_mae"]))

    results["variant_val_mae"] = variant_val_mae
    results["n_cols_per_variant"] = {v: int(len(models[v][2])) for v in models}

    # PRIMARY variant is selected on VALIDATION MAE only (never on test).
    primary = min((v for v in VARIANTS if v in models), key=lambda v: variant_val_mae[v])
    results["primary_variant"] = primary

    # ---------- baselines + Claim-A ceiling ----------
    base_preds, ceiling = build_baselines(
        S["A_val"]["X"], y_va, S["A_test"]["X"], y_te, SCALAR_SIGNALS, names, seed=seed)
    all_preds = {**base_preds, **preds_test}

    metrics = {}
    for k, p in all_preds.items():
        metrics[k] = full_metrics(p, y_te, g_te, n_boot=n_boot, seed=seed)
    ceiling_metrics = {k: full_metrics(p, y_te, g_te, n_boot=n_boot, seed=seed)
                       for k, p in ceiling.items()}
    results["metrics"] = metrics
    results["ceiling_metrics"] = ceiling_metrics

    # best ceiling = lowest MAE among optimistically-fitted monotone maps
    best_ceil_name = min(ceiling, key=lambda k: mae(ceiling[k], y_te))
    results["best_ceiling"] = best_ceil_name

    # ---------- decision rules (pre-registered) ----------
    gfull = preds_test.get(f"g_{primary}")
    dec = {"primary_variant": primary}
    if gfull is not None:
        dec["vs_constant"] = paired_diff_ci(mae, g_te, gfull, base_preds["B1_fixed_horizon"],
                                            y_te, n_boot=n_boot, seed=seed)
        dec["vs_best_ceiling"] = paired_diff_ci(mae, g_te, gfull, ceiling[best_ceil_name],
                                                y_te, n_boot=n_boot, seed=seed)
        dec["spearman"] = bootstrap_ci(spearman, g_te, gfull, y_te, n_boot=n_boot, seed=seed)
        strict = preds_test.get("g_strict_apriori")
        if strict is not None:
            dec["strict_vs_constant"] = paired_diff_ci(
                mae, g_te, strict, base_preds["B1_fixed_horizon"], y_te, n_boot=n_boot, seed=seed)
            dec["strict_vs_best_ceiling"] = paired_diff_ci(
                mae, g_te, strict, ceiling[best_ceil_name], y_te, n_boot=n_boot, seed=seed)
        hand = preds_test.get("g_hand_actions_only")
        if hand is not None:
            dec["vs_hand_actions"] = paired_diff_ci(mae, g_te, gfull, hand, y_te,
                                                    n_boot=n_boot, seed=seed)
    results["decisions"] = dec

    # ---------- H3: OOD transfer ----------
    ood = {}
    for key in ["B_test", "C_test", "D_test"]:
        sp = S[key]
        rec = {"H_star": degeneracy_report(sp["y"])}
        for vname in {primary, "strict_apriori"}:
            if vname not in models:
                continue
            m, std, cols = models[vname]
            xt = torch.as_tensor(std(sp["X"][:, cols]), device=device)
            mean, _ = m.predict(xt)
            rec[vname] = {
                "pred_mean": float(mean.mean()),
                **{k: v for k, v in full_metrics(mean, sp["y"], sp["groups"],
                                                 n_boot=max(200, n_boot // 4), seed=seed).items()},
            }
        ood[key] = rec
    results["ood"] = ood

    # ---------- predictions for figures ----------
    sig_cols = {f"sig__{lab}": S["A_test"]["X"][:, col[f]]
                for lab, f in SCALAR_SIGNALS.items() if f in col}
    np.savez_compressed(out_dir / f"predictions_seed{seed}.npz",
                        y_test=y_te, groups_test=g_te, E_test=S["A_test"]["E"], **sig_cols,
                        **{f"pred__{k}": v for k, v in all_preds.items()},
                        **{f"ceil__{k}": v for k, v in ceiling.items()},
                        **{f"ood__{k}__y": S[k]["y"] for k in ["B_test", "C_test", "D_test"]},
                        **{f"ood__{k}__E": S[k]["E"] for k in ["B_test", "C_test", "D_test"]})
    for key in ["B_test", "C_test", "D_test"]:
        m, std, cols = models[primary]
        xt = torch.as_tensor(std(S[key]["X"][:, cols]), device=device)
        mean, _ = m.predict(xt)
        np.save(out_dir / f"ood_pred_{key}_seed{seed}.npy", mean)

    json.dump(results, open(out_dir / f"analysis_seed{seed}.json", "w"), indent=2, default=float)
    return results
