"""Phase 8: baselines, including the critical Claim-A ceiling.

Baselines 1-8 from PREREGISTRATION.md sec. 9. Every baseline returns a point estimate on
the same scale as H* using the conditional-mean convention, so metrics are comparable.

The Claim-A ceiling is the key control: for each single uncertainty scalar we fit an
isotonic regression u -> H* **on the test set itself**. That is deliberately optimistic --
it is the best possible performance of ANY monotone function of that scalar. If g_phi does
not beat it, the finding reduces to the already-published weak claim.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.horizon.rih import H_MAX


def isotonic_fit_predict(u_fit, y_fit, u_eval):
    """Monotone (either direction) map from a scalar signal to H*."""
    iso = IsotonicRegression(out_of_bounds="clip", increasing="auto")
    iso.fit(np.asarray(u_fit, float).ravel(), np.asarray(y_fit, float).ravel())
    return iso.predict(np.asarray(u_eval, float).ravel())


def constant_baseline(y_fit, n_eval):
    """Baseline 1: fixed horizon = median H* on the fitting split."""
    return np.full(n_eval, float(np.median(y_fit)))


def random_baseline(y_fit, n_eval, seed=0):
    """Baseline 2: uniform random horizon over the observed range."""
    rng = np.random.default_rng(seed)
    return rng.uniform(float(np.min(y_fit)), float(np.max(y_fit)), size=n_eval)


def oracle_baseline(y_true):
    """Baseline 8: uses the future. Upper bound only."""
    return np.asarray(y_true, float)


def build_baselines(feats_val, y_val, feats_test, y_test, scalar_map, names, seed=0):
    """Returns {baseline_name: predictions_on_test} plus the Claim-A ceiling dict."""
    col = {n: i for i, n in enumerate(names)}
    preds: dict[str, np.ndarray] = {}
    n_test = len(y_test)

    preds["B1_fixed_horizon"] = constant_baseline(y_val, n_test)
    preds["B2_random_horizon"] = random_baseline(y_val, n_test, seed=seed)

    # Baselines 3-7: single-signal, isotonic-calibrated on VALIDATION (honest).
    for label, feat in scalar_map.items():
        if feat not in col:
            continue
        j = col[feat]
        preds[f"iso_val::{label}"] = isotonic_fit_predict(
            feats_val[:, j], y_val, feats_test[:, j]
        )

    preds["B8_oracle"] = oracle_baseline(y_test)

    # Claim-A ceiling: isotonic fitted ON TEST (optimistic upper bound for monotone maps).
    ceiling: dict[str, np.ndarray] = {}
    for label, feat in scalar_map.items():
        if feat not in col:
            continue
        j = col[feat]
        ceiling[label] = isotonic_fit_predict(feats_test[:, j], y_test, feats_test[:, j])
    return preds, ceiling


def raw_signal_direction(u, y):
    """Sign of the monotone relationship, for reporting Claim A (correlation)."""
    from scipy import stats as sps
    if np.std(u) == 0:
        return 0.0
    return float(sps.spearmanr(u, y).statistic)
