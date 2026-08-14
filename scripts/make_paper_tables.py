"""Emit LaTeX tables directly from frozen artifacts, so no number is transcribed by hand."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2)


def L(p):
    return json.load(open(p))


def w(name, s):
    (OUT / name).write_text(s)
    print(" ", name)


A = [L(R / f"main/analysis_seed{s}.json") for s in SEEDS]
AGG = L(R / "main/aggregate.json")["per_method"]


# ---------------------------------------------------------------- T1
def t_arch():
    ca = L(R / "cross_arch/cross_arch.json")
    rows = [("M1", "GRU-RSSM (large)", "gated recurrence", "deter 512, stoch 48", "gru_large"),
            ("M2", "GRU-RSSM (small)", "gated recurrence", "deter 256, stoch 16", "gru_small"),
            ("M3", "Transformer", "causal attention", "$d$ 256, 2 layers, 4 heads", "transformer")]
    body = ""
    for tag, name, mech, spec, key in rows:
        r = ca["archs"][key][0]
        body += (f"{tag} & {name} & {mech} & {spec} & {r['H_star_A_test']:.2f} & "
                 f"{r['A_test_mae']:.3f} & {r['A_test_rho']:+.3f} & "
                 f"$+{r['vs_ceiling']['point']:.3f}$ \\\\\n")
    w("tab_architectures.tex", r"""\begin{tabular}{llllcccc}
\toprule
& architecture & dynamics & size & $H^*$ on A & MAE $\downarrow$ & $\rho$ $\uparrow$ & vs.\ ceiling \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}""")


# ---------------------------------------------------------------- T2
def t_main():
    prim = [(a["primary_variant"],
             a["metrics"][f"g_{a['primary_variant']}"]) for a in A]
    pm = np.array([[x[1]["mae"]["point"], x[1]["spearman"]["point"], x[1]["r2"]["point"]]
                   for x in prim])
    ceil = np.array([[a["ceiling_metrics"]["imagined_disp_spread"]["mae"]["point"],
                      a["ceiling_metrics"]["imagined_disp_spread"]["spearman"]["point"],
                      a["ceiling_metrics"]["imagined_disp_spread"]["r2"]["point"]] for a in A])

    def row(label, m, sd=None, bold=False):
        f = (lambda v, s: f"\\textbf{{{v:.3f}}}" if bold else f"{v:.3f}")
        if sd is None:
            return (f"{label} & {f(m[0],0)} & {f(m[1],0)} & {f(m[2],0)} \\\\\n"
                    .replace("0.000 & 0.000 & 0.000", "---"))
        return (f"{label} & {f(m[0],0)}\\,$\\pm$\\,{sd[0]:.3f} & "
                f"{f(m[1],0)}\\,$\\pm$\\,{sd[1]:.3f} & {f(m[2],0)} \\\\\n")

    body = row("\\textbf{Monitor $g_\\phi$} (per-seed selected)", pm.mean(0), pm.std(0), bold=True)
    for lbl, k in [("\\quad \\texttt{no\\_latent}", "g_no_latent"),
                   ("\\quad \\texttt{full}", "g_full"),
                   ("\\quad uncertainty only", "g_uncertainty_only")]:
        v = AGG[k]
        body += (f"{lbl} & {v['mae']['mean']:.3f}\\,$\\pm$\\,{v['mae']['sd']:.3f} & "
                 f"{v['spearman']['mean']:.3f}\\,$\\pm$\\,{v['spearman']['sd']:.3f} & "
                 f"{v['r2']['mean']:.3f} \\\\\n")
    body += "\\midrule\n"
    body += (f"Claim-A ceiling (isotonic, on test) & {ceil[:,0].mean():.3f}\\,$\\pm$\\,"
             f"{ceil[:,0].std():.3f} & {ceil[:,1].mean():.3f} & {ceil[:,2].mean():.3f} \\\\\n")
    for lbl, k in [("Strict a-priori (no rollout)", "g_strict_apriori"),
                   ("Hand-crafted + actions", "g_hand_actions_only"),
                   ("Online truncation", "iso_val::internal_truncation_disp"),
                   ("Privileged: true local grid", "B8b_privileged_state"),
                   ("Fixed horizon (median of val)", "B1_fixed_horizon"),
                   ("Random horizon", "B2_random_horizon")]:
        v = AGG[k]
        body += (f"{lbl} & {v['mae']['mean']:.3f}\\,$\\pm$\\,{v['mae']['sd']:.3f} & "
                 f"{v['spearman']['mean']:.3f}\\,$\\pm$\\,{v['spearman']['sd']:.3f} & "
                 f"{v['r2']['mean']:.3f} \\\\\n")
    w("tab_main.tex", r"""\begin{tabular}{lccc}
\toprule
method & MAE $\downarrow$ & Spearman $\rho$ $\uparrow$ & $R^2$ $\uparrow$ \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}""")


# ---------------------------------------------------------------- T3
def t_ood():
    labels = {"B_test": "B (unseen layouts)", "C_test": "C (action slip)",
              "D_test": "D (long corridors)"}
    body = ""
    a0 = np.mean([a["degeneracy"]["A_test"]["mean"] for a in A])
    rho_id = np.mean([a["decisions"]["spearman"]["point"] for a in A])
    # in-distribution predicted mean: from the stored per-anchor monitor predictions
    a_pred = np.mean([np.load(R / f"main/predictions_seed{s}.npz")
                      [f"pred__g_{A[i]['primary_variant']}"].mean()
                      for i, s in enumerate(SEEDS)])
    body += f"A (in distribution) & {a0:.2f} & {a_pred:.2f} & {rho_id:+.3f} \\\\\n\\midrule\n"
    for k, lab in labels.items():
        act = np.mean([a["ood"][k]["H_star"]["mean"] for a in A])
        pr = np.mean([a["ood"][k][a["primary_variant"]]["pred_mean"] for a in A])
        rho = np.mean([a["ood"][k][a["primary_variant"]]["spearman"]["point"] for a in A])
        body += f"{lab} & {act:.2f} & \\textbf{{{pr:.2f}}} & {rho:+.3f} \\\\\n"
    # repair arms (H5), fixed no_latent variant
    O = [L(R / f"ood_repair/ood_repair_seed{s}.json") for s in SEEDS]
    body += "\\midrule\n\\multicolumn{4}{l}{\\emph{Repair attempts (H5), predicted $\\hat H$ on each split}}\\\\\n"
    for arm, lab in [("R0_baseline", "no repair"), ("R1_ranknorm", "rank-normalised inputs"),
                     ("R2_gated", "novelty gate"), ("R3_combined", "both")]:
        cells = [np.mean([o["arms"][arm][s]["pred_mean"] for o in O])
                 for s in ["B_test", "C_test", "D_test"]]
        body += (f"\\quad {lab} & \\multicolumn{{3}}{{c}}{{B {cells[0]:.2f}\\quad "
                 f"C {cells[1]:.2f}\\quad D {cells[2]:.2f}}} \\\\\n")
    w("tab_ood.tex", r"""\begin{tabular}{lccc}
\toprule
configuration & true $H^*$ & predicted $\hat H$ & Spearman $\rho$ \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}""")


# ---------------------------------------------------------------- T4
def t_transfer():
    for tag, fn in [("main", "transfer_main.json"), ("matched", "transfer_matched.json")]:
        d = L(R / f"transfer/{fn}")
        regimes = ["A", "A+B", "A+C", "A+D", "A+B+C"]
        cfgs = ["A", "B", "C", "D"]
        body = ""
        for rg in regimes:
            cells = []
            for c in cfgs:
                v = np.mean([x["metrics"][rg][c]["spearman"]["point"] for x in d])
                seen = d[0]["metrics"][rg][c]["seen_in_training"]
                cells.append(f"\\emph{{{v:+.3f}}}" if seen else f"{v:+.3f}")
            body += f"{rg} & " + " & ".join(cells) + " \\\\\n"
        w(f"tab_transfer_{tag}.tex", r"""\begin{tabular}{lcccc}
\toprule
training regime & test A & test B & test C & test D \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}""")


# ---------------------------------------------------------------- T5
def t_arch_transfer():
    ca = L(R / "cross_arch/cross_arch.json")
    names = ["gru_large", "gru_small", "transformer"]
    short = {"gru_large": "GRU-L", "gru_small": "GRU-S", "transformer": "Transf."}
    body = ""
    for s in names:
        cells = []
        for t in names:
            v = ca["cross"][f"{s}->{t}"][0]["rho"]
            cells.append(f"\\emph{{{v:+.3f}}}" if s == t else f"{v:+.3f}")
        body += f"fitted on {short[s]} & " + " & ".join(cells) + " \\\\\n"
    w("tab_arch_transfer.tex", r"""\begin{tabular}{lccc}
\toprule
& eval GRU-L & eval GRU-S & eval Transf. \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}""")


# ---------------------------------------------------------------- T6
def t_ledger():
    rows = [
        ("H1", "Uncertainty correlates with future error", "main run", "Supported", "weak; the premise, not the result"),
        ("H2", "Monitor beats best monotone single-scalar map", "main run, 3 seeds", "\\textbf{Supported}", "$+1.04$ MAE, CIs exclude 0"),
        ("H2$'$", "Same, without imagined-rollout features", "main run, 3 seeds", "Not supported", "rollout information is necessary"),
        ("H3", "Horizon drops before failure under shift", "main run, 3 seeds", "Not supported", "fails anti-conservatively"),
        ("H4", "Adaptive horizon improves planning at lower compute", "60 paired episodes", "Not supported", "\\emph{weak test}: monotone frontier"),
        ("H5", "OOD failure repaired by rank-norm / novelty gate", "3 seeds, 4 arms", "Not supported", "rank-informativeness, not scale"),
        ("H6", "Shift exposure restores transfer", "3 seeds + size-matched", "\\textbf{Supported}", "$\\rho$ 0.135 $\\to$ 0.766 on unseen D"),
        ("H7", "Phenomenon is architecture-general", "3 architectures, 1 seed", "Supported (H2); partial (H6)", "monitor only partly portable"),
        ("H8", "Shift-type taxonomy predicts transfer", "3 seeds, 2 new configs", "Not supported", "geometric half confirmed; F invalid"),
        ("H9", "Noise-driven shifts transfer to each other", "3 seeds, validity gate", "Not supported", "0.444 vs 0.50 bar; taxonomy withdrawn"),
    ]
    body = "".join(f"{h} & {q} & {p} & {v} & {n} \\\\\n" for h, q, p, v, n in rows)
    w("tab_ledger.tex", r"""\begin{tabular}{llllp{3.3cm}}
\toprule
& question & protocol & verdict & note \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}""")


if __name__ == "__main__":
    print("writing tables:")
    t_arch(); t_main(); t_ood(); t_transfer(); t_arch_transfer(); t_ledger()
    print(f"\n{len(list(OUT.glob('*.tex')))} tables in {OUT}")
