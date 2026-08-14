# AUDIT — existing manuscript vs. frozen experimental artifacts

Performed before any edit to `PAPER.md`. Every number below was read from stored artifacts
(`results/**/*.json`), not recomputed from features. Where the manuscript and the artifacts
disagree, the artifact is treated as ground truth.

**Verdict of the audit: the scientific story is sound and supported. Three numerical
inconsistencies must be fixed before submission, one of which is serious. No claim needs to be
withdrawn; several need tightening.**

---

## 1. Current paper structure

`PAPER.md` (350 lines, commit `0c4c4ab`): Abstract → 1 Introduction (+contributions) →
2 Related Work → 3 Problem Formulation → 4 Protocol → 5 Results (5.1–5.7) → 6 Limitations →
7 Conclusion → Appendix.

Adequate as a narrative; not yet ICML-shaped. Missing: a separate world-model/architecture
section, a mechanistic-analysis section (currently folded into 5.3/5.5), a discussion distinct
from the conclusion, reproducibility details, and any figures or numbered tables.

## 2. Main scientific claims

| # | Claim | Section |
|---|---|---|
| C1 | A learned monitor predicts `H*` and beats an optimistically-fitted monotone single-scalar ceiling | 5.1 |
| C2 | The monitor requires imagined-rollout information; strictly a-priori features fail | 5.1 |
| C3 | Under shift the monitor fails anti-conservatively | 5.2 |
| C4 | The failure is loss of rank-informativeness, not of scale | 5.3 |
| C5 | Exposure to a shifted distribution substantially restores transfer | 5.4 |
| C6 | Restoration is asymmetric across shift types | 5.5 |
| C7 | The core phenomenon is not recurrence-specific; monitor portability is partial | 5.6 |
| C8 | Adaptive planning did not help (weak test) | 5.7 |

## 3. Evidence supporting each claim

- **C1.** Per-seed primary monitor vs. optimistic ceiling: **+1.080 / +1.102 / +0.936 MAE**,
  bootstrap CI lower bounds **0.978 / 1.010 / 0.854** (all exclude 0). Ceiling signal is
  `imagined_disp_spread` on all three seeds. In-distribution ECE 0.023–0.041.
- **C2.** `g_strict_apriori` vs. ceiling: **−0.464 / −0.630 / −0.735** (all negative, all seeds).
  Fixed-variant MAE 8.065 ± 0.087, ρ 0.372 ± 0.003 vs. ceiling MAE 7.456.
- **C3.** Main-run H3: actual `H*` B 7.75 / C 4.11 / D 5.11 against predictions 21.01 / 12.82 /
  19.97; ρ 0.219 / 0.271 / 0.165.
- **C4.** H5: all four arms fail. `dispspread_mean` mean rises OOD (0.517 → 0.986/1.455/0.971)
  while within-split ρ collapses (−0.446 → −0.111 / −0.191 / −0.089). Entropy moves the *wrong*
  way (−0.394 → −0.186 / −0.107 / **+0.001**).
- **C5.** `A+B+C → D`: MAE 16.768 → **3.584**, ρ 0.135 → **0.766**. Size-matched control:
  MAE 3.622, ρ 0.762.
- **C6.** Geometric→geometric 0.732–0.855; geometric→noise 0.311; noise→noise 0.444–0.468;
  noise→geometric 0.388–0.539. Carrier `free_ahead`: A +0.004, B +0.586, D +0.566, E +0.533,
  C +0.024, F2 −0.015, F3 −0.031.
- **C7.** All three architectures beat their own ceiling (+1.355 / +1.050 / +0.993, CI lower
  bounds 1.251 / 0.945 / 0.901). Cross-architecture ρ: diagonal 0.584/0.551/0.513;
  off-diagonal 0.207–0.391.
- **C8.** Best fixed H=5 success 0.933 ± 0.032; adaptive 0.517 ± 0.065 at 5.49× compute.

## 4. Evidence contradicting or limiting each claim

- **C1** — R² is modest (0.318 for the per-seed primary). The margin is ~1 MAE against a target
  with sd ≈ 10.3. 21% of A-test targets are right-censored at `H_max = 30`.
- **C3/C5** — measured on *different splits*: H3 uses the full OOD split; H6/H8/H9 use 20%
  test portions (~2.2k anchors). Baselines therefore differ (`A→D` 0.165 full vs. 0.135
  portion). Currently undisclosed in the manuscript.
- **C5** — `A+F2` (a near-in-distribution config) teaches nothing, so "exposure helps" holds
  only when the donor genuinely degrades the model. Also **does not replicate in the
  Transformer** (0.257 → 0.290).
- **C6** — rests on 4 donors / 4 recipients inside one environment family. `A+C → F3` = 0.444
  **misses** its pre-registered 0.50 bar.
- **C7** — **single seed, K=3** for all three architectures (`n_seeds` = 1 in `cross_arch.json`).
  No variance estimate; not statistically comparable to the 3-seed main run.
- **C8** — frontier is monotone (mean detour **0.13** cells), so no interior optimum exists.

## 5. H1–H9 verdicts (from frozen artifacts)

| H | Question | Verdict |
|---|---|---|
| H1 | Uncertainty correlates with future error | SUPPORTED (weak; expected) |
| H2 | Monitor beats best monotone single-scalar map | **SUPPORTED** (3/3 seeds, CIs exclude 0) |
| H2-strict | Same, without imagined-rollout features | **NOT SUPPORTED** (3/3 seeds negative) |
| H3 | Predicted horizon drops before failure under shift | **NOT SUPPORTED** (fails anti-conservatively) |
| H4 | Adaptive horizon improves planning at lower compute | NOT SUPPORTED (**weak test**) |
| H5 | OOD failure repairable by rank-norm / novelty gate | **NOT SUPPORTED** (3/3 seeds, all arms) |
| H6 | Shift exposure restores transfer | **SUPPORTED** (+ size-matched control) |
| H7 | Phenomenon is architecture-general | SUPPORTED for H2; **PARTIAL** for H6 (single seed) |
| H8 | Taxonomy predicts transfer to new configs | NOT SUPPORTED by frozen rule; geometric half confirmed (ρ +0.855), noise half untested (F invalid) |
| H9 | Noise-driven family transfers within itself | **NOT SUPPORTED** (0.444 vs 0.50 bar); taxonomy withdrawn |

## 6. Important deviations (from `DEVIATIONS.md`)

D1 env configs changed after a no-network geometry survey; D2 collection policy changed
(`P(moved|forward)` 0.985 → 0.48–0.67); D3 splits restructured to give the monitor its own
world-model-held-out split; D4 pilot test seeds moved outside the main test range; D5
privileged baseline + val-based variant selection added; D6 planning task made non-myopic and
adaptive charged for its probe rollout; D7 all reported numbers from Kaggle (P100, torch
2.5.1+cu121 — local hedge run terminated and deleted); D8 planning re-run, detour manipulation
still failed; D9 H5 pre-registered blind; D10 H6 size-matched control + new 60/20/20 partitions.

All D1–D6 occurred **before** any test evaluation. D7–D10 are post-main-run and scoped to
follow-ups.

## 7. Potential reviewer objections

| # | Objection | Status |
|---|---|---|
| R1 | Single environment family, symbolic obs, 2.8M params | Real. Disclose prominently; scope the claim. |
| R2 | 21% right-censoring at `H_max` inflates MAE, upper tail unidentified | Real. Must be stated with the number. |
| R3 | "Beats the ceiling by ~1 MAE" is small vs. sd 10.3 | Real. Report effect size honestly; ρ and R² alongside. |
| R4 | If it needs the rollout, how does it differ from online truncation? | Answerable: beats truncation by 1.99 MAE (8.384 vs 6.394 fixed-variant). |
| R5 | H7 single seed presented alongside 3-seed results | Must be labelled in every table caption. |
| R6 | H6's effect may be "learn to predict near-zero OOD" rather than genuine monitoring | Partly true; the Transformer result is the evidence. Must be stated, not buried. |
| R7 | Splits differ between H3 and H6/H8/H9 | **Currently undisclosed.** Must add. |
| R8 | Asymmetry from 4 donors could be severity, not shift type | Real alternative explanation; must be offered explicitly. |
| R9 | Isotonic ceiling fitted on test — is that a fair control? | It is deliberately *optimistic*; explain that it strengthens, not weakens, the comparison. |

## 8. Numerical inconsistencies (must fix)

**N1 — SERIOUS. §5.2 cites H5's R0 arm as if it were the main-run H3 result.**
Manuscript: B 21.56 / ρ 0.199; C 13.59 / 0.238; D 21.15 / 0.128.
Main-run H3 (`REPORT_main.md`): B **21.01** / **0.219**; C **12.82** / **0.271**;
D **19.97** / **0.165**.
The manuscript's figures are the H5 `R0_baseline` arm (fixed `no_latent` variant), a different
experiment. Both are valid measurements; presenting one as the other is not. **Fix:** use
main-run H3 in §5.2 and cite H5's R0 separately as a replication.

**N2 — Composite statistics in the headline row.** Manuscript reports the primary monitor as
MAE 6.42 ± 0.07, ρ 0.573 ± 0.005, R² 0.331. The per-seed primary values (`full`, `no_latent`,
`no_latent`) give **6.416 ± 0.087**, **ρ 0.573 ± 0.011**, **R² 0.318 ± 0.020**. The quoted ±
values and R² are the *fixed* `no_latent` row (6.394 ± 0.068, ρ 0.580 ± 0.005, R² 0.331). Means
from one computation, dispersions from another. **Fix:** report both rows explicitly — the
per-seed-selected primary with its own sd, and the fixed `no_latent` variant — and state that
variant selection differed by seed.

**N3 — Minor.** Abstract says the margin is "≈1.02 MAE"; the mean of 1.080/1.102/0.936 is
**1.039**. Planning detour reported as 0.1; stored value is **0.13**. §5.3's ρ_A for
`imagined_disp_spread` is quoted as −0.455 (H6 rank-stability table, 20% portions) while the
adjacent OOD values are H5's (full split); the two sources must not be mixed within one table.

## 9. Missing methodological details

1. World-model architecture never fully specified (encoder, GRU sizes, latent dims, heads).
2. Feature groups named but not defined; the 74-column shared subspace is never enumerated.
3. Training hyperparameters absent (6000 steps, batch 64, seq len 50, lr 3e-4, AdamW).
4. Monitor architecture/optimisation absent (MLP 256×2, 60 epochs, val-based selection).
5. Bootstrap procedure described only as "episode-clustered"; the clustering unit and resample
   count (2000) need stating in-method.
6. `H*` right-censoring never analysed, only mentioned.
7. Collection policy (forward-biased, `p_forward` 0.6, scripted subgoals 0.35) not described.
8. The 60/20/20 partition of OOD configs, and that test portions are evaluation-only in every
   regime, is stated in protocols but not in the paper.
9. Compute/runtime not reported (main run 4.4 h on P100; 24 world models total).

## 10. Claims currently too strong

| Location | Current | Recommended |
|---|---|---|
| Title | "Can a World Model Know When Its Imagination Will Break?" | Keep only as a rhetorical hook; the technical claim must be the monitor formulation. |
| Abstract | "restoration is asymmetric" | Add: rests on 4 donors, one environment family. |
| §5.4 | "transfer can be substantially restored" | Add the Transformer counterexample *in the same paragraph*. |
| §5.5 | "we identify the carrier" | `free_ahead` is *a* carrier in this domain; the general form is "an ID-useless, shift-dominant cue exists". |
| §5.6 | architecture table without seed caveat | Label single-seed in the caption, not only in prose. |
| §7 | "cannot be fixed by rescaling" | "was not fixed by the two repairs we tested". |

## 11. Figures/tables that should exist but don't

Currently the manuscript has **no figures** and only inline markdown tables. Existing PNGs
(`results/main_kaggle/figs/main/`) cover the main run only and are not paper-shaped.

Needed: (F1) conceptual overview + architecture; (F2) predicted vs. actual `H*` with the
ceiling; (F3) OOD anti-conservative failure; (F4) H6 transfer heatmap; (F5) cross-architecture
3×3; (F6) feature rank-stability (ID vs OOD usefulness); (F7) donor→recipient asymmetry.
Tables: architectures; main result; OOD + repair; transfer matrix; cross-architecture;
H1–H9 ledger.

All are derivable from existing artifacts — no new experiments required.

## 12. Recommended final paper structure

1 Introduction · 2 Related Work · 3 Problem Formulation (`H*`, monitor, Claim-A ceiling) ·
4 World Models and the Reliability Horizon (3 architectures) · 5 Experimental Setup
(protocol, splits, statistics, censoring) · 6 Results (6.1 core → 6.2 rollout necessity →
6.3 shift failure → 6.4 repairs fail → 6.5 exposure → 6.6 architecture → 6.7 asymmetry) ·
7 Mechanistic Analysis (scale vs rank-informativeness) · 8 Limitations · 9 Discussion ·
10 Conclusion · Appendices (A architecture/hyperparameters, B features, C full H1–H9,
D transfer matrices, E statistics/censoring, F deviations, G H4 planning, H validity gates).
