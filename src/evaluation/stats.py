"""Phase 13: statistics. Episode-clustered bootstrap, correlations, effect sizes."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

N_BOOT = 10_000


def _cluster_resample(rng, groups_idx: list[np.ndarray], n_boot: int):
    """Yield bootstrap index arrays formed by resampling whole episodes."""
    G = len(groups_idx)
    for _ in range(n_boot):
        pick = rng.integers(0, G, size=G)
        yield np.concatenate([groups_idx[p] for p in pick])


def group_index(groups: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(groups, kind="stable")
    gs = groups[order]
    bounds = np.flatnonzero(np.diff(gs)) + 1
    return np.split(order, bounds)


def bootstrap_ci(fn, groups: np.ndarray, *arrays, n_boot: int = N_BOOT, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """Episode-clustered bootstrap CI for a statistic `fn(*arrays)`."""
    rng = np.random.default_rng(seed)
    gi = group_index(groups)
    point = float(fn(*arrays))
    vals = np.empty(n_boot)
    for b, idx in enumerate(_cluster_resample(rng, gi, n_boot)):
        vals[b] = fn(*[a[idx] for a in arrays])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "lo": float(lo), "hi": float(hi), "se": float(vals.std())}


def paired_diff_ci(fn, groups: np.ndarray, a_pred, b_pred, y, n_boot: int = N_BOOT,
                   seed: int = 0, alpha: float = 0.05) -> dict:
    """CI for fn(b_pred, y) - fn(a_pred, y), paired on the same resampled episodes."""
    rng = np.random.default_rng(seed)
    gi = group_index(groups)
    point = float(fn(b_pred, y) - fn(a_pred, y))
    vals = np.empty(n_boot)
    for i, idx in enumerate(_cluster_resample(rng, gi, n_boot)):
        vals[i] = fn(b_pred[idx], y[idx]) - fn(a_pred[idx], y[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "p_gt0": float((vals <= 0).mean()), "se": float(vals.std())}


def mae(pred, y):
    return float(np.abs(np.asarray(pred) - np.asarray(y)).mean())


def rmse(pred, y):
    return float(np.sqrt(((np.asarray(pred) - np.asarray(y)) ** 2).mean()))


def r2(pred, y):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def spearman(pred, y):
    if np.std(pred) == 0 or np.std(y) == 0:
        return 0.0
    return float(sps.spearmanr(pred, y).statistic)


def pearson(pred, y):
    if np.std(pred) == 0 or np.std(y) == 0:
        return 0.0
    return float(sps.pearsonr(pred, y).statistic)


def kendall(pred, y):
    if np.std(pred) == 0 or np.std(y) == 0:
        return 0.0
    return float(sps.kendalltau(pred, y).statistic)


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((a.var(ddof=1) * (len(a) - 1)) + (b.var(ddof=1) * (len(b) - 1))) /
                (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else float("nan")


def full_metrics(pred, y, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    out = {}
    for name, fn in [("mae", mae), ("rmse", rmse), ("r2", r2),
                     ("spearman", spearman), ("pearson", pearson), ("kendall", kendall)]:
        out[name] = bootstrap_ci(fn, groups, np.asarray(pred), np.asarray(y),
                                 n_boot=n_boot, seed=seed)
    return out


def calibration_curve(prob_matrix: np.ndarray, y: np.ndarray, horizons: list[int]) -> dict:
    """Reliability of P(H* >= h): predicted probability vs empirical frequency."""
    out = {}
    for h in horizons:
        p = prob_matrix[:, h:].sum(1)          # predicted P(H* >= h)
        actual = (y >= h).astype(float)
        bins = np.clip((p * 10).astype(int), 0, 9)
        rows = []
        for b in range(10):
            m = bins == b
            if m.sum() >= 20:
                rows.append({"bin": b, "pred": float(p[m].mean()),
                             "emp": float(actual[m].mean()), "n": int(m.sum())})
        ece = sum(r["n"] * abs(r["pred"] - r["emp"]) for r in rows) / max(1, sum(r["n"] for r in rows))
        out[h] = {"bins": rows, "ece": float(ece)}
    return out
