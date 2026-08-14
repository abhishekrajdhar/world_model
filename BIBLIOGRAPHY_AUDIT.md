# Bibliography Verification Audit

Every recent preprint referenced in `docs/phase0_literature.md` was checked against its arXiv
abstract page. Nine identifiers were extracted; **all nine resolved and were verified**. Each
verified entry is now cited properly in the manuscript, and the paragraph directing readers to
the Phase-0 survey in lieu of citations has been removed.

## Verified and added (9/9)

| arXiv | Title (as published) | Authors | Submitted | BibTeX key |
|---|---|---|---|---|
| 2606.13092 | Certified World Models: Predictability Across Configuration, Horizon, and Resolution | Hongbo Wang | 11 Jun 2026 | `wang2026certified` |
| 2606.24946 | Conformal Orbit-Valid Trust Horizons for Equivariant World Models | Hongbo Wang | 23 Jun 2026 | `wang2026conformal` |
| 2608.00675 | Round-Trip Consistency: Bidirectional Diffusion Models Can Predict Their Own Rollout Errors | Alexander Scheinker | 1 Aug 2026 (v2 13 Aug) | `scheinker2026roundtrip` |
| 2604.25416 | Biased Dreams: Limitations to Epistemic Uncertainty Quantification in Latent **Dynamics** Models | Julia Berger, Bernd Frauenknecht, Sebastian Trimpe, Bastian Leibe | 28 Apr 2026 | `berger2026biased` |
| 2606.27326 | Hallucination in World Models **is Predictable and Preventable** | Nicklas Hansen, Xiaolong Wang | 25 Jun 2026 | `hansen2026hallucination` |
| 2606.27780 | Understanding Rollout Error in Graph World Models | Xinyuan Song, Zekun Cai | 26 Jun 2026 (v2 5 Jul) | `song2026rollout` |
| 2605.04709 | ELVIS: Ensemble-Calibrated Latent Imagination for Long-Horizon Visual MPC | Yurui Du, Pinhao Song, Yutong Hu, Renaud Detry | 6 May 2026 | `du2026elvis` |
| 2504.16680 | Uncertainty-Aware Robotic World Model Makes Offline Model-Based RL Work on Real Robots | Chenhao Li, Andreas Krause, Marco Hutter | 23 Apr 2025 (v3 8 Jan 2026) | `li2025uncertainty` |
| 2206.02380 | Adaptive Rollout Length for Model-Based RL Using Model-Free Deep RL | Abhinav Bhatia, Philip S. Thomas, Shlomo Zilberstein | 6 Jun 2022 | `bhatia2022adaptive` |

## Could not be verified

None. No entry was omitted for lack of verification, and no entry was fabricated.

## Metadata corrections against the Phase-0 survey

The survey recorded two titles inexactly. Both are corrected in `references.bib`:

1. **2604.25416** — survey: "…in Latent *Space* Models"; actual: "…in Latent **Dynamics** Models".
2. **2606.27326** — survey: "Hallucination in World Models: Predictable & Preventable";
   actual: "Hallucination in World Models **is Predictable and Preventable**".

The survey also listed only first authors or none; full author lists are now recorded.

## Scientific / positioning claims corrected against primary sources

Four claims did not survive contact with the primary papers and were changed. None affects any
experimental result, hypothesis or conclusion.

**C1 — ELVIS mischaracterised as adaptive-horizon MPC.**
The Phase-0 survey described ELVIS as "adaptive-horizon MPC with real-time confidence
thresholds". The abstract describes something different: an ensemble of latent critics defines
a UCB score that gates a **time-varying $\lambda$**, trading bootstrapping against look-ahead,
alongside Gaussian-mixture planning for multi-modal futures. That adapts the *value estimator*,
not a predicted horizon. Related Work now says so explicitly, which if anything *widens* our
novelty gap rather than narrowing it.

**C2 — We had implied we contradict Berger et al.**
The draft said predictive entropy "matches the attractor-collapse account… ensemble spread does
not, so that account is signal-specific rather than general." Berger et al.'s claim is that
uncertainty *declines along an extended rollout* as it is drawn into well-modelled attractor
regions. We measure across-distribution means and within-distribution rank correlations — a
different axis. §7 now states that our observations "neither confirm nor contradict that
account", and that testing it would require measuring uncertainty along the rollout, which we
did not do. This is the most substantive correction in the pass: we were claiming to bear on a
result we had not actually tested.

**C3 — "Invertible model" was wrong for Round-Trip Consistency.**
Scheinker trains a *bidirectional conditional latent diffusion model with a direction flag*,
not an invertible one. Corrected to "a model trained to run in both directions". The domains
are magnetohydrodynamics, astrophysical mixing layers and face video — no acting agent — which
we now state, since it is the sharper distinction.

**C4 — The conformal comparison was underspecified, and is stronger than we had it.**
Wang (2606.24946) reports **zero anti-conservative violations** across the stable audits, with a
median certified-to-measured ratio of 0.67. Our headline negative is an *anti-conservative*
failure. Related Work now makes the contrast explicit: the failure we document is precisely
what a conformal construction is designed to exclude, and our monitor carries no such
guarantee. This sharpens the positioning honestly rather than softening it.

## Positioning strengthened, not weakened

One verified paper improves our framing. \citet{hansen2026hallucination} conclude that
hallucination in large visual world models is "inherently a data coverage issue", predictable
from coverage signals and preventable by coverage-aware collection. Our H6 result — that the
monitor's shift failure is repaired by *exposure to the right kind of failure* rather than by
better use of the uncertainty signal already present — is convergent evidence from a different
direction and at a very different scale. Related Work now says this. It is a genuine
independent-convergence argument that we did not previously have.

## Novelty boundary after verification

Unchanged. No verified paper reports a learned, per-state horizon monitor evaluated against an
optimistic monotone control and characterised across shifts and architectures. The two
certificate papers are analytic, symmetry-dependent and orbit-level; Round-Trip Consistency is
per-rollout, needs a backward pass, and has no agent; the graph-world-model analysis is
model-level and post hoc; ELVIS adapts a value estimator. The nearest negative result
(Berger et al.) concerns a mechanism we did not measure.

## Final citation audit

- Every `\cite` key resolves to `references.bib`: **yes** (0 unresolved).
- Every relevant 2026 preprint carries a verified primary-source citation: **yes** (9/9).
- Fabricated entries: **none**.
- Orphaned entries (defined but never cited): **none**.
- Related Work accurately represents each cited paper: **yes**, after corrections C1–C4.
- Pointer to `docs/phase0_literature.md` as a citation substitute: **removed**.
- Structural check (`scripts/check_paper.py`): **0 problems**.
