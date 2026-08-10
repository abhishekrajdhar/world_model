"""Publication-quality figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluation.stats import calibration_curve, r2, spearman  # noqa: E402
from src.horizon.rih import H_MAX  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.autolayout": True,
})
C = {"main": "#2b6cb0", "alt": "#c05621", "grey": "#718096",
     "good": "#2f855a", "bad": "#9b2c2c", "acc": "#6b46c1"}


def fig1_architecture(out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axis("off")
    boxes = [("obs $o_t$\n5x5x3", 0.03), ("encoder\n$e_t$", 0.19), ("GRU\n$h_t$", 0.35),
             ("prior $p(z_t|h_t)$\nposterior $q(z_t|h_t,e_t)$", 0.53),
             ("heads\ndir, moved,\ncarry, door", 0.74), ("displacement\nby integration", 0.90)]
    for label, x in boxes:
        ax.add_patch(plt.Rectangle((x, 0.42), 0.12, 0.34, fc="white", ec=C["main"], lw=1.2,
                                   transform=ax.transAxes, clip_on=False))
        ax.text(x + 0.06, 0.59, label, ha="center", va="center", fontsize=7.2,
                transform=ax.transAxes)
        if x < 0.88:
            ax.annotate("", xy=(x + 0.145, 0.59), xytext=(x + 0.122, 0.59),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1))
    ax.annotate("", xy=(0.41, 0.42), xytext=(0.41, 0.22), xycoords=ax.transAxes,
                textcoords=ax.transAxes, arrowprops=dict(arrowstyle="<-", color=C["alt"], lw=1))
    ax.text(0.42, 0.16, "action $a_t$  (imagination: no observation input)",
            fontsize=7, color=C["alt"], transform=ax.transAxes)
    ax.text(0.5, 0.92, "Latent world model: heads are size-agnostic state factors; "
                       "position is reconstructed by integrating $moved$ along $dir$",
            ha="center", fontsize=7.5, transform=ax.transAxes)
    fig.savefig(out / "fig1_architecture.png", bbox_inches="tight")
    plt.close(fig)


def fig2_error_vs_horizon(E_by_split: dict[str, np.ndarray], out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    hs = np.arange(1, H_MAX + 1)
    for name, E in E_by_split.items():
        m = E.mean(0)
        se = E.std(0) / np.sqrt(len(E))
        axes[0].plot(hs, m, label=name, lw=1.4)
        axes[0].fill_between(hs, m - 1.96 * se, m + 1.96 * se, alpha=0.15)
        axes[1].plot(hs, (E == 0).mean(0), label=name, lw=1.4)
    axes[0].set_xlabel("imagination horizon $H$"); axes[0].set_ylabel("$E_H$ (cells)")
    axes[0].set_title("Prediction error vs horizon")
    axes[1].set_xlabel("imagination horizon $H$"); axes[1].set_ylabel("$P(E_H = 0)$")
    axes[1].set_title("Probability imagination is still exact")
    axes[1].legend(fontsize=7)
    fig.savefig(out / "fig2_error_vs_horizon.png", bbox_inches="tight")
    plt.close(fig)


def fig3_uncertainty_vs_error(u: np.ndarray, E: np.ndarray, uname: str, out: Path,
                              horizons=(1, 5, 10, 20)):
    fig, axes = plt.subplots(1, len(horizons), figsize=(2.0 * len(horizons), 2.4), sharey=False)
    qs = np.quantile(u, np.linspace(0, 1, 11))
    qs[-1] += 1e-9
    for ax, h in zip(np.atleast_1d(axes), horizons):
        e = E[:, h - 1]
        binned = np.digitize(u, qs[1:-1])
        xs, ys, lo, hi = [], [], [], []
        for b in range(10):
            m = binned == b
            if m.sum() < 10:
                continue
            xs.append(u[m].mean()); ys.append(e[m].mean())
            s = e[m].std() / np.sqrt(m.sum())
            lo.append(ys[-1] - 1.96 * s); hi.append(ys[-1] + 1.96 * s)
        ax.plot(xs, ys, "o-", color=C["main"], ms=3, lw=1.2)
        ax.fill_between(xs, lo, hi, alpha=0.2, color=C["main"])
        ax.set_title(f"$H={h}$  " + r"$\rho$=" + f"{spearman(u, e):.2f}", fontsize=8)
        ax.set_xlabel(uname, fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("$E_H$ (cells)")
    fig.suptitle("Claim A: uncertainty at $t$ vs future imagination error", fontsize=9)
    fig.savefig(out / "fig3_uncertainty_vs_error.png", bbox_inches="tight")
    plt.close(fig)


def fig4_predicted_vs_actual(pred: np.ndarray, y: np.ndarray, out: Path,
                             extra: dict[str, np.ndarray] | None = None):
    """THE headline figure: predicted RIH vs actual RIH, per state."""
    ncol = 1 + (len(extra) if extra else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(3.1 * ncol, 3.0), squeeze=False)
    panels = [("$g_\\phi$ (full)", pred)] + list((extra or {}).items())
    for ax, (name, p) in zip(axes[0], panels):
        jitter = np.random.default_rng(0).uniform(-0.35, 0.35, len(y))
        ax.scatter(p, y + jitter, s=3, alpha=0.16, color=C["main"], edgecolors="none")
        # binned conditional mean
        bins = np.linspace(p.min(), p.max(), 13)
        idx = np.digitize(p, bins[1:-1])
        bx, by = [], []
        for b in range(12):
            m = idx == b
            if m.sum() >= 15:
                bx.append(p[m].mean()); by.append(y[m].mean())
        ax.plot(bx, by, "o-", color=C["alt"], ms=4, lw=1.6, label="conditional mean")
        lim = [0, H_MAX]
        ax.plot(lim, lim, "--", color=C["grey"], lw=1, label="$y=x$")
        ax.set_xlim(-0.5, H_MAX + 0.5); ax.set_ylim(-1, H_MAX + 1)
        ax.set_xlabel("predicted RIH  $\\hat{H}^*$")
        ax.set_title(f"{name}\n$R^2$={r2(p, y):.3f}  " + r"$\rho$=" + f"{spearman(p, y):.3f}",
                     fontsize=8.5)
    axes[0][0].set_ylabel("actual RIH  $H^*$")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.savefig(out / "fig4_predicted_vs_actual_RIH.png", bbox_inches="tight")
    plt.close(fig)


def fig5_calibration(prob: np.ndarray, y: np.ndarray, out: Path, horizons=(3, 5, 10, 20)):
    cal = calibration_curve(prob, y, list(horizons))
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], "--", color=C["grey"], lw=1, label="perfect")
    for h, colr in zip(horizons, [C["main"], C["alt"], C["good"], C["acc"]]):
        rows = cal[h]["bins"]
        if not rows:
            continue
        ax.plot([r["pred"] for r in rows], [r["emp"] for r in rows], "o-", ms=3.5, lw=1.3,
                color=colr, label=f"$H\\geq{h}$ (ECE={cal[h]['ece']:.3f})")
    ax.set_xlabel("predicted $P(H^* \\geq h)$"); ax.set_ylabel("empirical frequency")
    ax.set_title("Calibration of the horizon predictor", fontsize=9)
    ax.legend(fontsize=7)
    fig.savefig(out / "fig5_calibration.png", bbox_inches="tight")
    plt.close(fig)
    return cal


def fig7_ood(records: dict[str, dict], out: Path):
    """records[name] = {'y': actual H*, 'pred': predicted H*}"""
    names = list(records)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    xs = np.arange(len(names))
    act = [records[n]["y"].mean() for n in names]
    prd = [records[n]["pred"].mean() for n in names]
    act_se = [records[n]["y"].std() / np.sqrt(len(records[n]["y"])) for n in names]
    prd_se = [records[n]["pred"].std() / np.sqrt(len(records[n]["pred"])) for n in names]
    axes[0].bar(xs - 0.2, act, 0.4, yerr=act_se, label="actual $H^*$", color=C["main"], capsize=3)
    axes[0].bar(xs + 0.2, prd, 0.4, yerr=prd_se, label="predicted $\\hat{H}^*$",
                color=C["alt"], capsize=3)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(names, fontsize=7.5)
    axes[0].set_ylabel("horizon (steps)"); axes[0].legend(fontsize=7)
    axes[0].set_title("Does the predictor lower its horizon under shift?", fontsize=9)
    for n, colr in zip(names, [C["main"], C["alt"], C["good"], C["acc"], C["bad"]]):
        r = records[n]
        axes[1].scatter(r["pred"], r["y"] + np.random.default_rng(0).uniform(-.3, .3, len(r["y"])),
                        s=3, alpha=0.12, color=colr, edgecolors="none")
        axes[1].plot([], [], "o", color=colr, ms=4,
                     label=f"{n} " + r"$\rho$=" + f"{spearman(r['pred'], r['y']):.2f}")
    axes[1].plot([0, H_MAX], [0, H_MAX], "--", color=C["grey"], lw=1)
    axes[1].set_xlabel("predicted $\\hat{H}^*$"); axes[1].set_ylabel("actual $H^*$")
    axes[1].legend(fontsize=6.5); axes[1].set_title("Rank transfer under shift", fontsize=9)
    fig.savefig(out / "fig7_ood.png", bbox_inches="tight")
    plt.close(fig)


def fig8_failures(pred: np.ndarray, y: np.ndarray, E: np.ndarray, out: Path):
    """Overconfident and underconfident cases, and what their error curves look like."""
    resid = pred - y
    over = np.argsort(-resid)[:200]     # predicted high, actual low
    under = np.argsort(resid)[:200]     # predicted low, actual high
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    hs = np.arange(1, H_MAX + 1)
    for ax, idx, title, colr in [
        (axes[0], over, "Overconfident\n(pred high, actual low)", C["bad"]),
        (axes[1], under, "Underconfident\n(pred low, actual high)", C["good"])]:
        ax.plot(hs, E[idx].mean(0), color=colr, lw=1.6, label="mean $E_H$")
        for j in idx[:25]:
            ax.plot(hs, E[j], color=colr, lw=0.4, alpha=0.25)
        ax.set_title(title, fontsize=8.5); ax.set_xlabel("$H$"); ax.set_ylabel("$E_H$")
    axes[2].hist(resid, bins=40, color=C["main"], alpha=0.85)
    axes[2].axvline(0, color=C["grey"], ls="--", lw=1)
    axes[2].set_title(f"Residual $\\hat{{H}}^*-H^*$\nmean={resid.mean():.2f} "
                      f"sd={resid.std():.2f}", fontsize=8.5)
    axes[2].set_xlabel("residual (steps)"); axes[2].set_ylabel("count")
    fig.savefig(out / "fig8_failures.png", bbox_inches="tight")
    plt.close(fig)
    return {"n_overconfident_gt5": int((resid > 5).sum()),
            "n_underconfident_lt-5": int((resid < -5).sum()),
            "frac_overconfident_gt5": float((resid > 5).mean()),
            "frac_underconfident_lt-5": float((resid < -5).mean())}


def fig6_planning(results: dict, out: Path):
    """Compute-performance frontier: adaptive vs fixed horizons."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    fixed = {k: v for k, v in results.items() if k.startswith("fixed_")}
    order = sorted(fixed, key=lambda k: int(k.split("_")[1]))
    xs = [fixed[k]["imagined_states_mean"] for k in order]
    ys = [fixed[k]["success_rate"] for k in order]
    yerr = [fixed[k].get("success_se", 0) for k in order]
    axes[0].errorbar(xs, ys, yerr=yerr, marker="o", color=C["main"], lw=1.3, ms=4,
                     label="fixed horizon", capsize=3)
    for k, x, y in zip(order, xs, ys):
        axes[0].annotate(k.split("_")[1], (x, y), fontsize=6.5,
                         textcoords="offset points", xytext=(3, 4))
    if "adaptive" in results:
        a = results["adaptive"]
        axes[0].errorbar([a["imagined_states_mean"]], [a["success_rate"]],
                         yerr=[a.get("success_se", 0)], marker="*", ms=13, color=C["alt"],
                         label="adaptive $\\hat{H}^*$", capsize=3)
    axes[0].set_xlabel("imagined states per decision (compute)")
    axes[0].set_ylabel("success rate")
    axes[0].set_title("Compute-performance frontier", fontsize=9)
    axes[0].legend(fontsize=7)

    labels = order + (["adaptive"] if "adaptive" in results else [])
    vals = [results[k]["steps_to_goal_mean"] for k in labels]
    errs = [results[k].get("steps_se", 0) for k in labels]
    cols = [C["main"]] * len(order) + ([C["alt"]] if "adaptive" in results else [])
    axes[1].bar(range(len(labels)), vals, yerr=errs, color=cols, capsize=3)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels([l.replace("fixed_", "H=") for l in labels], fontsize=7, rotation=30)
    axes[1].set_ylabel("steps to goal (lower better)")
    axes[1].set_title("Planning performance", fontsize=9)
    fig.savefig(out / "fig6_planning.png", bbox_inches="tight")
    plt.close(fig)
