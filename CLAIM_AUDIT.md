# CLAIM AUDIT

Every scientific claim intended for the Abstract, Introduction, Discussion or Conclusion,
checked against the frozen artifacts. **Strength** is one of:

- **A** — pre-registered, tested, supported on all seeds, CI excludes the null
- **B** — supported but single-seed, or from a follow-up protocol, or with a caveat
- **C** — descriptive/interpretive; consistent with data but not itself a tested hypothesis
- **D** — unsupported as previously worded; must be weakened or cut

| # | Claim | Evidence | Pre-reg? | Strength | Recommended wording |
|---|---|---|---|---|---|
| 1 | A learned monitor predicts `H*` better than the best monotone function of any single uncertainty scalar | vs-ceiling +1.080/+1.102/+0.936, CI lo 0.978/1.010/0.854 (3/3 seeds) | H2, yes | **A** | "improves on an optimistically-fitted monotone single-scalar ceiling by 1.04 MAE on average (per-seed CI lower bounds 0.85–1.01)" — use 1.04, not 1.02 |
| 2 | The monitor also beats a constant predictor | +2.505/+2.631/+2.541, CI lo ≥ 2.335 | H2, yes | **A** | keep |
| 3 | Rollout information is necessary | strict a-priori vs ceiling −0.464/−0.630/−0.735 (3/3 negative) | H2-strict, yes | **A** | "a monitor restricted to time-`t` features does not beat the ceiling on any seed" |
| 4 | The monitor beats online truncation | 8.384 vs 6.394 (fixed `no_latent`) = 1.99 MAE | baseline, yes | **A** | quote as fixed-variant comparison, state the variant |
| 5 | A privileged predictor with the true local grid does worse | 8.628 / ρ 0.226 vs 6.394 / ρ 0.580 | H, yes (B8b) | **A** | keep; note it is an empirical comparison, not a proven ceiling |
| 6 | Under shift the monitor is anti-conservative | H3: actual 7.75/4.11/5.11 vs predicted 21.01/12.82/19.97 | H3, yes | **A** | **use main-run H3 numbers**, not H5's R0 arm (see AUDIT N1) |
| 7 | Ranking quality collapses OOD | ρ 0.219/0.271/0.165 vs 0.573 ID | H3, yes | **A** | keep |
| 8 | The OOD failure is not a scaling problem | H5: all 4 arms fail 3/3 seeds | H5, yes | **A** | "the two repairs we tested — rank-normalisation and novelty gating — do not recover it" |
| 9 | Features retain aggregate separation but lose within-distribution rank-informativeness | `dispspread` mean 0.517→0.986/1.455/0.971 while within-split ρ −0.446→−0.111/−0.191/−0.089 | diagnostic, reported | **B** | present as the mechanistic account this experiment supports, labelled as a *diagnostic* not a pre-registered test |
| 10 | Ensemble disagreement does *not* show attractor collapse; entropy does | spread rises OOD; entropy ρ −0.394→+0.001 on D | diagnostic | **B** | keep, attribute per-signal; do not generalise to "ensembles are fine" |
| 11 | Exposure to a shifted distribution substantially restores transfer | A+B+C→D: MAE 16.768→3.584, ρ 0.135→0.766 | H6, yes | **A** | keep the "can restore" phrasing |
| 12 | The effect is not explained by sample count alone | size-matched: MAE 3.622, ρ 0.762 at equal anchors | H6, yes | **A** | "the size-matched control indicates the effect is not explained by sample count alone"; do **not** claim diversity is the sole cause |
| 13 | A second shift adds little | A+B→D 0.771 vs A+B+C→D 0.766 (matched: 0.759 vs 0.762) | H6, descriptive | **C** | "the marginal value of a second shift was small in our setting" |
| 14 | Restoration requires the donor to genuinely degrade the model | A+F2→B 0.226, →D 0.087 (≤ baseline); F2 `H*` 12.72 | H9 validity gate | **B** | keep; it is the control that makes the point |
| 15 | Transfer is asymmetric: geometric→geometric ≫ geometric→noise | 0.732–0.855 vs 0.311 | H8/H9 | **B** | keep, state it rests on 4 donors in one environment family |
| 16 | Noise donors transfer broadly but weakly | 0.388–0.539 to geometric, 0.444–0.468 to noise | H8/H9 | **B** | keep |
| 17 | `free_ahead` is the carrier of geometric transfer | ρ: A +0.004, B +0.586, D +0.566, E +0.533; C +0.024, F2 −0.015, F3 −0.031 | H8 P6, pre-registered | **A** | "a cue that is uninformative in distribution and dominant under geometric shift"; the *general* form is C-strength |
| 18 | Severity, not shift type, could explain the asymmetry | not tested; F2/C/F3 differ in severity as well as type | no | **D→C** | must be offered explicitly as an untested alternative explanation |
| 19 | The epistemic/aleatoric taxonomy holds | H8 NOT SUPPORTED; H9 `A+C→F3` 0.444 < 0.50 bar | H8/H9, yes | **D** | **withdrawn.** Must not reappear; use "geometric / noise-driven" as descriptive labels only |
| 20 | The core result is not specific to recurrent world models | all 3 architectures beat own ceiling (+1.355/+1.050/+0.993, CI lo ≥ 0.901) | H7, yes | **B** | "replicates in a non-recurrent Transformer"; **label single-seed, K=3** |
| 21 | Monitor portability decays across architectures | diag 0.584/0.551/0.513; within-family 0.356–0.391; cross-family 0.207–0.362 | H7, yes | **B** | keep; single-seed; do not claim statistical separation between 0.356 and 0.391 |
| 22 | The restoration effect is architecture-dependent | Transformer A→D 0.257 vs A+B→D 0.290 | H7, yes | **B** | keep |
| 23 | Restoration scales with how badly the base model degrades | Transformer `H*` on D 8.55, zero-frac 3.8% vs GRUs ≈5.1 / ≈29% | H7, descriptive | **C** | "consistent with"; it is an interpretation of a single contrast, not a tested relationship |
| 24 | Adaptive planning did not help | 0.517 vs best fixed 0.933 at 5.49× compute | H4, yes | **A** | keep, and label **weak test** — frontier monotone, mean detour 0.13 |
| 25 | The monitor is well calibrated in distribution | ECE 0.023/0.030/0.041/0.036 | reported | **A** | specify these are `P(H* ≥ h)` for h ∈ {3,5,10,20} |
| 26 | "World models know when they are wrong" | — | no | **D** | **cut.** Replace with the monitor formulation throughout |
| 27 | "Training diversity solves OOD uncertainty" | — | no | **D** | **cut.** See claim 12 wording |
| 28 | Results generalise beyond MiniGrid / symbolic obs / 2.8M params | not tested | no | **D** | must be scoped explicitly in Limitations and never implied elsewhere |
| 29 | `H*` distribution is non-degenerate on A | mean 15.52–15.73, IQR 20–21, ceiling 20.7–21.7% | pre-registered gate | **A** | report the 21% censoring alongside |

## Wording changes required before submission

1. Replace every instance of "know(s) when it is wrong" with monitor language (claim 26).
2. Abstract margin: **1.04**, not 1.02 (claim 1).
3. §OOD table: use main-run H3 values 21.01 / 12.82 / 19.97 and ρ 0.219 / 0.271 / 0.165 (claim 6).
4. Headline row: report per-seed-selected primary as **6.416 ± 0.087 / ρ 0.573 ± 0.011 /
   R² 0.318 ± 0.020**, and the fixed `no_latent` variant separately as
   **6.394 ± 0.068 / 0.580 ± 0.005 / 0.331**. State that the selected variant differed by seed.
5. Every architecture table/caption: "single seed, K = 3".
6. Every planning mention: "weak test".
7. Add the severity confound as an explicit alternative explanation (claim 18).
8. Add a sentence disclosing that H3 uses full OOD splits while H6/H8/H9 use 20% test portions,
   so baselines differ between sections (AUDIT R7).
9. Planning detour: **0.13**, not 0.1.
10. Never reintroduce epistemic/aleatoric as an established taxonomy (claim 19).

## Claims deliberately *not* made

- No causal claim that diversity per se produces transfer (only that sample count does not
  explain it).
- No claim that the monitor is useful for planning (H4 was a weak test and failed).
- No claim about pixel-based, continuous-control, or large-scale world models.
- No claim that ensemble disagreement is generally reliable OOD — only that it did not collapse
  in the direction prior work predicted, on these configurations.
