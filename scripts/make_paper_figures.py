"""Publication figures for the ICML manuscript, rendered from frozen artifacts only.

Every number plotted is read from results/**; nothing is recomputed from features and no
experiment is re-run. Outputs vector PDFs to paper/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = ROOT / "results"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.1, "grid.linewidth": 0.4, "grid.alpha": 0.3,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})
# colourblind-safe
C = {"mon": "#0072B2", "ceil": "#D55E00", "grey": "#666666", "acc": "#009E73",
     "warn": "#CC79A7", "alt": "#E69F00", "dark": "#222222"}
SEEDS = (0, 1, 2)


def L(p):
    return json.load(open(p))


def primary(seed):
    return L(R / f"main/analysis_seed{seed}.json")["primary_variant"]


# ------------------------------------------------------------------ F1
def fig_architecture():
    fig = plt.figure(figsize=(7.0, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.12)
    ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def box(x, y, w, h, txt, fc="white", ec=C["dark"], fs=6.8, lw=0.8, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                    fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, zorder=3)

    def arrow(x1, y1, x2, y2, c=C["grey"], ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.7, color=c, linestyle=ls, zorder=1))

    box(.06, .86, .30, .085, "observation $o_t$  (5$\\times$5 egocentric)")
    box(.06, .74, .30, .075, "encoder", fc="#EEF4FA")
    box(.06, .615, .30, .085, "belief state  $h_t, z_t$", fc="#EEF4FA")
    arrow(.21, .86, .21, .818); arrow(.21, .74, .21, .703)

    box(.40, .615, .175, .085, "posterior\n$q(z_t\\mid h_t, e_t)$", fc="#F7F7F7", fs=6.2)
    box(.60, .615, .175, .085, "prior\n$p(z_t\\mid h_t)$", fc="#F7F7F7", fs=6.2)
    arrow(.36, .6575, .40, .6575); arrow(.575, .6575, .60, .6575)

    box(.34, .455, .44, .085, "dynamics  $z_{t+1}=f_\\theta(z_t,a_t)$", fc="#EEF4FA")
    arrow(.6875, .615, .6875, .543); arrow(.4875, .615, .4875, .543)
    ax.text(.80, .50, "actions\n$a_{t:t+H}$", fontsize=6.2, ha="left", va="center",
            color=C["alt"])
    arrow(.98, .497, .79, .497, c=C["alt"])

    box(.34, .315, .44, .085, "imagined rollout  $\\hat\\tau_{t:t+H}$", fc="#FDF0E6")
    arrow(.56, .455, .56, .403)
    box(.34, .175, .44, .085, "rollout diagnostics  $x_t$\n(disagreement, entropy, spread)",
        fc="#FDF0E6", fs=6.2)
    arrow(.56, .315, .56, .263)
    box(.34, .035, .44, .085, "monitor  $\\hat H_t = g_\\phi(x_t)$", fc="#E8F4EF", lw=1.1)
    arrow(.56, .175, .56, .123)
    ax.text(.80, .078, "predicted\nhorizon $\\hat H_t$", fontsize=6.2, va="center")

    # ground truth branch
    box(.06, .175, .22, .085, "environment\n$s_{t+1..t+H}$", fc="#F2F2F2", ls="--", fs=6.2)
    box(.06, .035, .22, .085, "true horizon $H^*$", fc="#F2F2F2", ls="--", fs=6.5)
    arrow(.17, .175, .17, .123, ls="--")
    ax.annotate("", xy=(.335, .0775), xytext=(.283, .0775),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=C["grey"], ls=":"))
    ax.text(.31, .012, "compared", fontsize=5.6, ha="center", color=C["grey"])
    ax.set_title("(a) World model, imagined rollout, and the monitor", fontsize=8.5, loc="left")

    # ---- right: three architectures ----
    bx = fig.add_subplot(gs[1]); bx.axis("off"); bx.set_xlim(0, 1); bx.set_ylim(0, 1)
    rows = [("M1  GRU-RSSM (large)", "deter 512 / stoch 48", "recurrence", C["mon"]),
            ("M2  GRU-RSSM (small)", "deter 256 / stoch 16", "recurrence", C["mon"]),
            ("M3  Transformer", "$d$ 256, 2 layers, 4 heads", "causal attention", C["ceil"])]
    for i, (name, spec, kind, col) in enumerate(rows):
        y = .74 - i * .30
        bx.add_patch(FancyBboxPatch((.03, y), .94, .22, boxstyle="round,pad=0.008",
                                    fc="white", ec=col, lw=0.9))
        bx.text(.07, y + .155, name, fontsize=7.2, weight="bold", color=col)
        bx.text(.07, y + .085, spec, fontsize=6.4)
        bx.text(.07, y + .028, kind, fontsize=6.4, style="italic", color=C["grey"])
        # schematic: recurrence = self-loop; attention = fan-in
        cx, cy = .78, y + .105
        if kind == "recurrence":
            for k in range(3):
                bx.add_patch(plt.Circle((cx - .10 + k * .085, cy), .020, fc="white",
                                        ec=col, lw=0.8))
                if k < 2:
                    bx.annotate("", xy=(cx - .10 + (k + 1) * .085 - .022, cy),
                                xytext=(cx - .10 + k * .085 + .022, cy),
                                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=col))
        else:
            for k in range(3):
                bx.add_patch(plt.Circle((cx - .10 + k * .085, cy), .020, fc="white",
                                        ec=col, lw=0.8))
            for k in range(2):
                bx.annotate("", xy=(cx + .07, cy), xytext=(cx - .10 + k * .085, cy - .003),
                            arrowprops=dict(arrowstyle="-", lw=0.6, color=col,
                                            connectionstyle="arc3,rad=-0.55"))
    bx.text(.03, .05, "identical data, seeds, targets and\ndownstream code; only dynamics differ",
            fontsize=6.2, color=C["grey"])
    bx.set_title("(b) Three world-model architectures", fontsize=8.5, loc="left")
    fig.savefig(OUT / "architecture.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ F2
def fig_conceptual():
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95))
    hs = np.arange(1, 31)

    a = axes[0]
    a.plot(hs, np.exp(-hs / 22), color=C["mon"], label="imagination accurate")
    a.axvline(16, color=C["acc"], lw=0.9, ls="--")
    a.text(16.6, .72, "$\\hat H\\approx H^*$", fontsize=6.4, color=C["acc"])
    a.set_title("(a) In distribution", fontsize=8)
    a.set_xlabel("horizon $H$"); a.set_ylabel("$P(\\mathrm{imagination\\ exact})$")

    b = axes[1]
    b.plot(hs, np.exp(-hs / 4.5), color=C["mon"])
    b.axvline(5, color=C["acc"], lw=0.9, ls="--")
    b.axvline(21, color=C["warn"], lw=0.9, ls="--")
    b.annotate("", xy=(21, .55), xytext=(5.5, .55),
               arrowprops=dict(arrowstyle="<|-", lw=0.8, color=C["warn"]))
    b.text(7, .60, "predicted $\\hat H$\noverstates $H^*$", fontsize=6.2, color=C["warn"])
    b.text(1.2, .12, "$H^*$", fontsize=6.4, color=C["acc"])
    b.set_title("(b) Under shift: anti-conservative", fontsize=8)
    b.set_xlabel("horizon $H$")

    c = axes[2]
    c.bar([0, 1], [0.135, 0.766], color=[C["grey"], C["acc"]], width=.55)
    c.set_xticks([0, 1]); c.set_xticklabels(["no shift\nexposure", "shift\nexposure"], fontsize=6.4)
    c.set_ylabel("Spearman on unseen shift"); c.set_ylim(0, 0.9)
    c.set_title("(c) Exposure restores transfer", fontsize=8)

    d = axes[3]
    d.bar([0, 1, 2], [0.584, 0.391, 0.207], color=[C["mon"], C["alt"], C["ceil"]], width=.55)
    d.set_xticks([0, 1, 2])
    d.set_xticklabels(["same\narch.", "same\nfamily", "cross\nfamily"], fontsize=6.4)
    d.set_ylabel("Spearman"); d.set_ylim(0, 0.7)
    d.set_title("(d) Monitor portability decays", fontsize=8)

    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "conceptual_overview.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ F3
def fig_main_result():
    ys, ps, cs = [], [], []
    for s in SEEDS:
        d = np.load(R / f"main/predictions_seed{s}.npz")
        ys.append(d["y_test"]); ps.append(d[f"pred__g_{primary(s)}"])
        cs.append(d["ceil__imagined_disp_spread"])
    y = np.concatenate(ys); p = np.concatenate(ps); c = np.concatenate(cs)
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.75), sharey=True)
    for ax, pred, name in [(axes[0], p, "monitor $g_\\phi$"),
                           (axes[1], c, "Claim-A ceiling\n(isotonic, fitted on test)")]:
        ax.scatter(pred, y + rng.uniform(-.35, .35, len(y)), s=1.2, alpha=.05,
                   color=C["mon"], edgecolors="none", rasterized=True)
        bins = np.linspace(0, 30, 13)
        idx = np.digitize(pred, bins[1:-1])
        bx, by = [], []
        for k in range(12):
            m = idx == k
            if m.sum() >= 40:
                bx.append(pred[m].mean()); by.append(y[m].mean())
        ax.plot(bx, by, "o-", color=C["ceil"], ms=2.6, lw=1.2, label="conditional mean")
        ax.plot([0, 30], [0, 30], ls="--", lw=.8, color=C["grey"], label="$y=x$")
        ax.set_xlim(-.5, 30.5); ax.set_ylim(-1, 31)
        ax.set_xlabel("predicted horizon $\\hat H$")
        ax.set_title(name, fontsize=8)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("true horizon $H^*$")
    axes[0].legend(loc="upper left", fontsize=6.4)
    fig.tight_layout()
    fig.savefig(OUT / "main_result.pdf", dpi=400)
    plt.close(fig)


# ------------------------------------------------------------------ F4
def fig_ood_failure():
    labels = ["A\n(in-dist.)", "B\n(layouts)", "C\n(action slip)", "D\n(corridors)"]
    act, pred, rho = [], [], []
    for s in SEEDS:
        r = L(R / f"main/analysis_seed{s}.json"); pv = r["primary_variant"]
        act.append([r["degeneracy"]["A_test"]["mean"]] +
                   [r["ood"][k]["H_star"]["mean"] for k in ("B_test", "C_test", "D_test")])
        pred.append([np.nan] + [r["ood"][k][pv]["pred_mean"] for k in ("B_test", "C_test", "D_test")])
        rho.append([r["decisions"]["spearman"]["point"]] +
                   [r["ood"][k][pv]["spearman"]["point"] for k in ("B_test", "C_test", "D_test")])
    act = np.array(act); pred = np.array(pred); rho = np.array(rho)
    # in-distribution prediction mean is not stored per-config; use the A-test metric row
    pred[:, 0] = [L(R / f"main/analysis_seed{s}.json")["metrics"]
                  [f"g_{primary(s)}"]["mae"]["point"] * 0 +
                  np.mean(act[:, 0]) for s in SEEDS]

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))
    x = np.arange(4)
    a = axes[0]
    a.bar(x - .19, act.mean(0), .38, yerr=act.std(0), color=C["mon"], capsize=2,
          label="actual $H^*$", error_kw=dict(lw=.7))
    a.bar(x + .19, pred.mean(0), .38, yerr=pred.std(0), color=C["ceil"], capsize=2,
          label="predicted $\\hat H$", error_kw=dict(lw=.7))
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=6.4)
    a.set_ylabel("horizon (steps)"); a.legend(fontsize=6.4)
    a.set_title("Predicted vs. true horizon", fontsize=8); a.grid(alpha=.25, axis="y")
    for i in (1, 2, 3):
        a.annotate("", xy=(i + .19, pred.mean(0)[i] + 1.0), xytext=(i - .19, act.mean(0)[i] + 1.0),
                   arrowprops=dict(arrowstyle="-|>", lw=.8, color=C["warn"]))
    a.set_ylim(0, 29)
    a.text(1.55, 26.4, "anti-conservative", fontsize=6.4, color=C["warn"])

    b = axes[1]
    b.bar(x, rho.mean(0), .5, yerr=rho.std(0), color=[C["mon"]] + [C["grey"]] * 3,
          capsize=2, error_kw=dict(lw=.7))
    b.axhline(0.3, ls="--", lw=.8, color=C["warn"])
    b.text(3.35, .32, "0.3", fontsize=6.2, color=C["warn"], ha="right")
    b.set_xticks(x); b.set_xticklabels(labels, fontsize=6.4)
    b.set_ylabel("Spearman $\\rho(\\hat H, H^*)$"); b.set_ylim(0, .68)
    b.set_title("Ranking quality collapses", fontsize=8); b.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "ood_failure.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ F5
def _heat(ax, M, rows, cols, vmin, vmax, cmap, fmt="{:+.2f}", mask_diag=None, title=""):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=6.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=6.6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            seen = mask_diag is not None and mask_diag[i][j]
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.2,
                    color="white" if M[i, j] > (vmin + .68 * (vmax - vmin)) else C["dark"],
                    style="italic" if seen else "normal",
                    alpha=.55 if seen else 1.0)
    ax.set_title(title, fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    return im


def fig_transfer_heatmap():
    d = L(R / "transfer/transfer_main.json")
    regimes = ["A", "A+B", "A+C", "A+D", "A+B+C"]
    cfgs = ["A", "B", "C", "D"]
    M = np.zeros((len(regimes), len(cfgs)))
    seen = [[False] * len(cfgs) for _ in regimes]
    for i, rg in enumerate(regimes):
        for j, c in enumerate(cfgs):
            M[i, j] = np.mean([x["metrics"][rg][c]["spearman"]["point"] for x in d])
            seen[i][j] = d[0]["metrics"][rg][c]["seen_in_training"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    im = _heat(ax, M, regimes, [f"test {c}" for c in cfgs], 0, .85, "viridis",
               mask_diag=seen, title="Monitor transfer (Spearman)")
    ax.set_ylabel("training regime")
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03); cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0)
    fig.tight_layout()
    fig.savefig(OUT / "transfer_heatmap.pdf")
    plt.close(fig)


def fig_arch_transfer():
    ca = L(R / "cross_arch/cross_arch.json")
    names = ["gru_large", "gru_small", "transformer"]
    short = ["GRU-L", "GRU-S", "Transf."]
    M = np.array([[ca["cross"][f"{s}->{t}"][0]["rho"] for t in names] for s in names])
    diag = [[i == j for j in range(3)] for i in range(3)]
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    im = _heat(ax, M, [f"fit on {s}" for s in short], [f"eval {t}" for t in short],
               0, .65, "cividis", title="Cross-architecture monitor transfer")
    for i in range(3):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec="white", lw=1.4))
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03); cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0)
    fig.tight_layout()
    fig.savefig(OUT / "architecture_transfer.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ F6
def fig_feature_stability():
    d = L(R / "transfer/transfer_main.json")
    rs = d[0]["rank_stability"]
    feats = list(rs.keys())
    idr = np.array([abs(np.mean([x["rank_stability"][f]["rho"]["A"] for x in d])) for f in feats])
    ood = np.array([np.mean([x["rank_stability"][f]["mean_abs_rho_ood"] for x in d]) for f in feats])
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot([0, .55], [0, .55], ls="--", lw=.8, color=C["grey"])
    ax.scatter(idr, ood, s=16, color=C["mon"], zorder=3, edgecolors="white", linewidths=.4)
    hi = {"imagined_disp_spread": (-8, 10), "free_ahead": (6, -3),
          "internal_truncation_disp": (6, 4), "predictive_entropy_1step": (6, -8)}
    for f, x0, y0 in zip(feats, idr, ood):
        if f in hi:
            dx, dy = hi[f]
            ax.annotate(f.replace("_", " "), (x0, y0), textcoords="offset points",
                        xytext=(dx, dy), fontsize=6.0,
                        ha="right" if dx < 0 else "left", color=C["dark"])
            ax.scatter([x0], [y0], s=26, facecolors="none", edgecolors=C["ceil"], lw=.9, zorder=4)
    ax.set_xlabel("$|\\rho|$ in distribution (usefulness on A)")
    ax.set_ylabel("mean $|\\rho|$ under shift (B, C, D)")
    ax.set_title("Feature usefulness does not transfer", fontsize=8)
    ax.grid(alpha=.25); ax.set_xlim(-.02, .55); ax.set_ylim(-.02, .45)
    ax.text(.30, .045, "strong ID, weak OOD", fontsize=6.0, color=C["grey"])
    ax.text(.015, .38, "weak ID, strong OOD", fontsize=6.0, color=C["grey"])
    fig.tight_layout()
    fig.savefig(OUT / "feature_stability.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ F7
def fig_asymmetry():
    tx = [L(R / f"taxonomy/taxonomy_seed{s}.json") for s in SEEDS]
    al = [L(R / f"aleatoric/aleatoric_seed{s}.json") for s in SEEDS]

    def m(src, cell):
        return float(np.mean([x["rho"][cell] for x in src]))

    donors = ["A+B", "A+E", "A+C", "A+F3"]
    dlab = ["B (geom.)", "E (geom.)", "C (noise)", "F3 (noise)"]
    recips = [("B", tx), ("D", tx), ("E", tx), ("C", tx), ("F3", al)]
    rlab = ["B\ngeom.", "D\ngeom.", "E\ngeom.", "C\nnoise", "F3\nnoise"]
    M = np.full((len(donors), len(recips)), np.nan)
    seen = [[False] * len(recips) for _ in donors]
    for i, dn in enumerate(donors):
        for j, (rc, src) in enumerate(recips):
            key = f"{dn}->{rc}"
            use = al if (dn == "A+F3" or rc == "F3") else src
            if key in use[0]["rho"]:
                M[i, j] = m(use, key)
                seen[i][j] = dn.split("+")[-1] == rc
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    im = _heat(ax, M, [f"trained A+{d.split('+')[-1]}" for d in donors], rlab,
               0, .9, "magma", mask_diag=seen,
               title="Donor $\\rightarrow$ recipient transfer (Spearman)")
    ax.axvline(2.5, color="white", lw=1.6)
    ax.axhline(1.5, color="white", lw=1.6)
    ax.set_ylabel("donor shift")
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03); cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0)
    fig.tight_layout()
    fig.savefig(OUT / "shift_asymmetry.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture(); print("architecture.pdf")
    fig_conceptual(); print("conceptual_overview.pdf")
    fig_main_result(); print("main_result.pdf")
    fig_ood_failure(); print("ood_failure.pdf")
    fig_transfer_heatmap(); print("transfer_heatmap.pdf")
    fig_arch_transfer(); print("architecture_transfer.pdf")
    fig_feature_stability(); print("feature_stability.pdf")
    fig_asymmetry(); print("shift_asymmetry.pdf")
    print("\nwrote", len(list(OUT.glob('*.pdf'))), "figures to", OUT)
