# Results — Reliable Imagination Horizon

Main run: Kaggle, 3 training seeds x 5-member ensembles, 15 world models, 4.4 h wall clock.
Auto-generated tables: [results/REPORT_main.md](results/REPORT_main.md).
Frozen protocol: [PREREGISTRATION.md](PREREGISTRATION.md) (commit `05cf014`).

---

## 1. Executive summary

A world-model agent **can** estimate its own reliable imagination horizon from information
available at time `t`, and it does so **better than any monotone function of any single
uncertainty scalar** — which is the part that goes beyond what the literature already shows.
The result holds on all three seeds with non-overlapping bootstrap CIs.

Three findings cut against the hypothesis and are, in our view, the more interesting half:

1. **The signal lives in the imagined rollout, not in the state at time `t`.** A predictor
   restricted to strictly a-priori features (belief state, one-step uncertainty, planned
   actions, observation statistics — no imagination) **fails** to beat the Claim-A ceiling on
   every seed. What works is reading the *internal disagreement of the imagined trajectory*.
2. **The horizon predictor does not survive distribution shift, and fails in the dangerous
   direction.** Out of distribution it predicts *longer* horizons while the true horizon
   collapses (config B: actual 7.8, predicted 20.0). It is most confident exactly where the
   world model is worst.
3. **Adaptive-horizon planning is worse and more expensive** than the best fixed horizon
   (−0.417 success at 5.49x compute). The same anti-conservative bias reappears: the planner
   selects H ≈ 26 where 3–5 is optimal. We flag this as a *weak* test, because the task's
   compute–performance frontier turned out to be monotone (§8.1).

## 2. Hypothesis verdict

| Claim | Verdict |
|---|---|
| **H1** — uncertainty correlates with future error (Claim A) | **SUPPORTED** (weak, as expected) |
| **H2** — learned predictor beats the best monotone single-scalar map (Claim B) | **SUPPORTED** |
| **H2-strict** — same, using only time-`t` features with no imagined rollout | **NOT SUPPORTED** |
| **H3** — predicted horizon drops before catastrophic error under shift | **NOT SUPPORTED** |
| **H4** — adaptive horizon improves planning at lower compute (Claim C) | **NOT SUPPORTED** (weak test — see §8.1) |
| **H5** — the OOD failure is repairable by scale-free inputs / a novelty gate | **NOT SUPPORTED** (see §8.2) |
| **H6** — exposure to diverse shifts restores cross-distribution horizon prediction | **SUPPORTED** (see §8.3) |
| **H7** — the result is a property of world models, not of one implementation | **SUPPORTED for H2; PARTIAL for H6** (see §8.4) |
| **H8** — the failure-mode taxonomy predicts transfer to new configs | **NOT SUPPORTED by the frozen rule**; epistemic half confirmed (ρ +0.855 on unseen E), aleatoric half untested (see §8.5) |

Overall, against the pre-registered decision rule in §11 of the protocol: **SUPPORTED**, with
the scope restricted to predictors allowed to inspect their own imagined rollout — and, after
H6, no longer restricted to in-distribution states, provided `g_phi` has seen at least one
distribution shift of the same failure family during training.

**The single most important result is H6 (§8.3).** The OOD failure is *not* intrinsic: one
shift in `g_phi`'s training set moves a never-seen config from MAE 16.77 / ρ +0.135 to
MAE 3.58 / ρ +0.766, and survives size-matching. But transfer is organised by **failure-mode
family** — epistemic shifts (unseen geometry) transfer to each other and not to the aleatoric
shift (action noise). The rank-stability analysis explains why: the predictor must switch from
a model-internal uncertainty cue that is strong in-distribution and collapses OOD, to an
observation-geometry cue that is useless in-distribution and dominant OOD.

## 3. Main quantitative results (A-test, mean ± sd over 3 seeds)

`H*` on A-test: mean 15.52, sd 10.29, IQR 20.0, 21.2 % at the `H_max=30` ceiling, 2.6 % at
zero. Non-degenerate by the pre-registered check (run on validation first).

| Method | MAE ↓ | RMSE ↓ | Spearman ↑ | R² ↑ |
|---|---|---|---|---|
| Oracle (uses the future) | 0.000 | 0.000 | 1.000 | 1.000 |
| **`g_φ` (primary, selected on val)** | **6.42 ± 0.07** | **8.37 ± 0.07** | **0.573 ± 0.005** | **0.331 ± 0.004** |
| `g_φ` full (latent + unc + imag + actions + hand) | 6.46 ± 0.08 | 8.54 ± 0.16 | 0.566 ± 0.005 | 0.304 ± 0.014 |
| `g_φ` uncertainty-only | 6.60 ± 0.11 | 8.55 ± 0.14 | 0.553 ± 0.012 | 0.301 ± 0.015 |
| **Claim-A ceiling** (isotonic on `imagined_disp_spread`, *fitted on test*) | 7.46 ± 0.13 | 9.05 ± 0.15 | 0.456 ± 0.012 | 0.218 ± 0.011 |
| `g_φ` **strict a-priori** (no imagined rollout) | 8.07 ± 0.09 | 9.60 ± 0.10 | 0.372 ± 0.003 | 0.119 ± 0.002 |
| Hand-crafted + actions (no world-model internals) | 8.27 ± 0.10 | 9.67 ± 0.09 | 0.320 ± 0.005 | 0.107 ± 0.001 |
| Online truncation (disp. spread ≥ 1) — MBPO/MACURA-style | 8.38 ± 0.08 | 9.73 ± 0.10 | 0.298 ± 0.011 | 0.096 ± 0.004 |
| **Privileged: true local 11×11 grid + ordered actions** | 8.63 ± 0.07 | 10.08 ± 0.12 | 0.226 ± 0.006 | 0.030 ± 0.006 |
| Baseline 1: fixed horizon (median) | 8.98 ± 0.10 | 10.28 ± 0.12 | 0.000 | −0.010 |
| Baseline 5: ensemble disagreement, 1-step (isotonic) | 9.03 ± 0.10 | 10.19 ± 0.09 | 0.103 ± 0.008 | 0.008 |
| Baseline 3: current-step error (isotonic) | 9.06 ± 0.11 | 10.24 ± 0.10 | 0.026 ± 0.003 | −0.002 |
| Baseline 2: random horizon | 11.00 ± 0.05 | 13.42 ± 0.04 | 0.001 | −0.721 |

Headline figure: [figures/fig4_predicted_vs_actual_RIH.png](results/main_kaggle/figs/main/fig4_predicted_vs_actual_RIH.png).
The conditional mean tracks `y = x` across the whole 0–30 range; **R² = 0.33, ρ = 0.57**.

## 4. Statistical results (paired, episode-clustered bootstrap, 2000 resamples)

Deltas are `MAE(baseline) − MAE(g_φ)`; positive means `g_φ` is better.

| Comparison | Seed 0 | Seed 1 | Seed 2 | CI lower bounds |
|---|---|---|---|---|
| vs constant (Baseline 1) | +2.505 | +2.631 | +2.541 | [2.335, 2.469, 2.384] |
| **vs best Claim-A ceiling** | **+1.080** | **+1.102** | **+0.936** | **[0.978, 1.010, 0.854]** |
| vs hand+actions | +1.817 | — | — | CI excludes 0 |
| **strict a-priori vs ceiling** | **−0.464** | **−0.630** | **−0.735** | all negative |
| Spearman(Ĥ*, H*) | 0.559 | 0.587 | 0.574 | [0.539, 0.567, 0.554] |

All three pre-registered conditions hold: (a) beats constant, (b) beats the optimistically
fitted monotone ceiling, (c) ρ > 0.5 with CI lower bound > 0.3.

**Calibration** (in-distribution): ECE 0.023 / 0.030 / 0.041 / 0.036 for `P(H* ≥ 3/5/10/20)`.
The predictive distribution is well calibrated where it was trained.

## 5. Figures

| Figure | Path |
|---|---|
| 1. Architecture | [fig1](results/main_kaggle/figs/main/fig1_architecture.png) |
| 2. Error vs horizon | [fig2](results/main_kaggle/figs/main/fig2_error_vs_horizon.png) |
| 3. Uncertainty vs future error (Claim A) | [fig3](results/main_kaggle/figs/main/fig3_uncertainty_vs_error.png) |
| **4. Predicted vs actual RIH** | [fig4](results/main_kaggle/figs/main/fig4_predicted_vs_actual_RIH.png) |
| 5. Calibration | [fig5](results/main_kaggle/figs/main/fig5_calibration.png) |
| 6. Planning frontier | [fig6](results/main_kaggle/figs/main/fig6_planning.png) |
| **7. OOD / distribution shift** | [fig7](results/main_kaggle/figs/main/fig7_ood.png) |
| 8. Failure cases | [fig8](results/main_kaggle/figs/main/fig8_failures.png) |

## 6. Failure cases

Residual `Ĥ* − H*` is roughly symmetric but heavy in both tails: **23.7 %** of states are
overconfident by more than 5 steps and **24.2 %** underconfident by more than 5.

- **Overconfident (predicts long, fails early).** Concentrated where the agent stands in an
  apparently open region whose far side it has not seen. Ensemble members agree — they share
  the same wrong prior that open space continues — so disagreement stays low while the true
  layout inserts a wall. This is precisely the attractor-collapse failure that
  [Biased Dreams](https://arxiv.org/abs/2604.25416) predicts for latent-space disagreement.
- **Underconfident (predicts short, stays accurate).** Dominated by states near doorways and
  corners, where members disagree about *where* the agent ends up but the planned action
  sequence happens not to visit the contested cells. The predictor cannot condition on the
  interaction between disagreement and the specific path, so it hedges.
- **The ceiling artefact.** 21 % of anchors have `H* = 30`, so the target is right-censored.
  `g_φ` correctly saturates there, but MAE is inflated by states whose true horizon exceeds
  the measurement window.

## 7. Ablations

**Ensemble size** (seed 0; no retraining — member subsets of the same ensemble):

| K | `g_φ` MAE | ρ | Claim-A ceiling MAE | Beats ceiling? |
|---|---|---|---|---|
| 1 | 6.91 | 0.533 | 8.90 | yes (+1.99) |
| 3 | 6.58 | 0.547 | 7.81 | yes (+1.23) |
| 5 | 6.50 | 0.559 | 7.58 | yes (+1.08) |

The effect **does not require an ensemble**: with K = 1 (no disagreement signal at all,
only single-model entropy and latent statistics) `g_φ` still beats its ceiling by the largest
margin. Ensembling improves absolute accuracy modestly and, notably, improves the *ceiling*
faster than it improves `g_φ` — the more members, the more of the signal a single scalar
captures.

**Feature groups** (which information carries the horizon signal):

| Variant | MAE | Reading |
|---|---|---|
| no_latent (unc + imag + actions + hand) | 6.39 | best |
| full | 6.46 | latent adds nothing, costs variance |
| uncertainty_only | 6.60 | uncertainty alone is nearly sufficient |
| strict a-priori (no imagination) | 8.07 | **imagination is necessary** |
| hand + actions | 8.27 | world-model internals matter |
| latent only | 8.45 | belief state alone is weak |
| actions only | 8.92 | ≈ constant |

**Seeds:** 3 independent repeats; sd of MAE ≤ 0.11 across all methods.
**Environments/stochasticity/shift:** configs A–D, §8.

Not run, for compute reasons (GPU quota): model-size and view-size/partial-observability
ablations. Both would require full retraining.

## 8. Distribution shift (Phase 12) — the informative failure

| Split | Actual `H*` | Predicted `Ĥ*` | Spearman |
|---|---|---|---|
| A (in-dist) | 15.52 | 15.55 | **0.573** |
| B (unseen layouts) | 7.75 | 20.01 | 0.13 |
| C (stochastic, slip 0.15) | 4.11 | 12.82 | 0.22 |
| D (long corridors) | 5.11 | 19.97 | 0.12 |

In distribution the predictor is nearly unbiased (15.55 vs 15.52). Under shift the true
horizon collapses by 2–4x while the prediction *rises*. Rank ordering also degrades below the
pre-registered 0.3 threshold on all three OOD configs.

This is the opposite of the safety property one would want, and it is a clean negative result
rather than a tuning failure: the predictor was never fitted on B/C/D, and nothing about its
inputs tells it that the world has changed. The uncertainty signals it relies on are
*relative* quantities calibrated to config A's difficulty distribution.

### 8.1 Claim C (H4): adaptive planning — NOT SUPPORTED

60 paired episodes per arm, identical layout and planner seeds
([fig6](results/main_kaggle/figs/main/fig6_planning.png)):

| Arm | Success | Steps to goal | Imagined states / decision | Mean H |
|---|---|---|---|---|
| fixed H=3 | 0.917 ± 0.036 | 28.1 | 288 | 3.0 |
| **fixed H=5 (best)** | **0.933 ± 0.032** | 34.5 | 480 | 5.0 |
| fixed H=10 | 0.767 ± 0.055 | 46.3 | 960 | 10.0 |
| fixed H=20 | 0.517 ± 0.065 | 58.1 | 1920 | 20.0 |
| fixed H=30 | 0.367 ± 0.062 | 63.7 | 2880 | 30.0 |
| **adaptive Ĥ\*** | 0.517 ± 0.065 | 60.7 | 2637 | 25.9 |

Adaptive is **worse and more expensive**: success delta −0.417 [−0.550, −0.300] versus the
best fixed arm, at **5.49x** its compute. The pre-registered H4 criterion fails on both legs.

**This is a weak test of H4, and we say so rather than claiming a result it cannot support.**
Success decreases *monotonically* with horizon, so the compute–performance frontier has **no
interior optimum** — there is nothing for an adaptive horizon to find. The intended fix (make
targets require detours) did not take: mean detour is **0.1 cells** and **57/60** layouts fell
through to the no-detour tier, so shortest-path ≈ Manhattan distance and the task stays
myopic. Longer random-shooting plans only compound model error.

The adaptive arm's specific failure is nonetheless informative: it selects **H ≈ 25.9** when
this planner wants 3–5. `g_φ` was fitted under the data-collection distribution, and MPC
visits a different state/action distribution, so the **same anti-conservative bias documented
in §8 reappears** — the predictor over-estimates its horizon exactly where doing so is
costly. H4's failure is therefore not independent of H3's; both trace to the predictor's
uncertainty features being calibrated to one difficulty distribution.

A genuine test of H4 needs a task whose fixed-horizon frontier is non-monotone (an interior
optimum), which this environment does not provide.

## 8.2 H5: attempting to repair the OOD failure — NOT SUPPORTED

Follow-up experiment, pre-registered blind in [PREREGISTRATION_H5.md](PREREGISTRATION_H5.md)
and run on the precomputed features (no retraining). Report:
[results/REPORT_H5.md](results/REPORT_H5.md).

Two repairs, neither fitted on B/C/D: **R1** rank-normalises the uncertainty features
(tests "the features are merely mis-scaled"); **R2** shrinks the prediction by a novelty gate
`exp(-λ·max(0, ν−τ))` whose constants come from A-val by a fixed rule; **R3** is both.

| Arm | A-test pred (actual 15.60) | B pred (actual 7.75) | C pred (actual 4.11) | D pred (actual 5.11) |
|---|---|---|---|---|
| R0 baseline | 15.68 | 21.56 | 13.59 | 21.15 |
| R1 rank-norm | 15.87 | 20.99 | **16.08** (worse) | 21.29 |
| R2 novelty-gated | 15.63 | 20.25 | 13.12 | 19.95 |
| R3 combined | 15.82 | 19.78 | 15.56 | 20.12 |

**Verdict: NOT SUPPORTED on all three seeds, for all three repair arms.** Criterion (1) —
the predicted horizon must *fall* out of distribution — fails everywhere: predictions stay
**above** the in-distribution value (≈20 vs 15.7) while the true horizon is 4–8. The gate did
behave as designed in-distribution (15.68 → 15.63, MAE 6.394 → 6.399, criterion (3) holds),
so the failure is not an artefact of a badly-calibrated gate; it simply never fires hard
enough.

### Why — and it is neither explanation we pre-registered

The diagnostic (per-split mean, and Spearman with `H*` computed *within* each split) shows
the two mechanisms are **mixed across features**, which is why a single fix cannot work:

| Feature | A-test | B | C | D |
|---|---|---|---|---|
| `dispspread_mean` | 0.53 (ρ **−0.46**) | 1.09 (ρ −0.11) | 1.50 (ρ −0.20) | 1.04 (ρ −0.09) |
| `ent_moved_mean` | 0.21 (ρ −0.40) | **0.15** (ρ −0.16) | 0.21 (ρ −0.11) | **0.15** (ρ +0.04) |
| `trunc_h_disp` | 24.4 (ρ +0.30) | 19.8 (ρ +0.10) | 16.6 (ρ +0.20) | 20.0 (ρ +0.09) |
| `latent_nn_dist` | 3.93 | 5.89 | 5.13 | 5.86 |

1. **Ensemble disagreement moves the *right* way** (spread 0.53 → 1.0–1.5 OOD), refuting a
   pure "attractor collapse" reading for that signal.
2. **Predictive entropy moves the *wrong* way** — it *falls* on B and D (0.21 → 0.15) exactly
   where the model is worst. This is the [Biased Dreams](https://arxiv.org/abs/2604.25416)
   failure, but confined to the entropy signal rather than the ensemble.
3. **The decisive problem is neither: every feature loses its per-state discriminative
   power.** `dispspread`'s within-split correlation with `H*` collapses from **−0.46** to
   −0.09…−0.20. The signal still says "things are harder here" on average, but it no longer
   ranks *which* states will fail — which is precisely what a per-state horizon needs.

That explains both failures directly. **R1 fails** because rank-normalising preserves an
ordering that has itself become uninformative (and on C it destroys the magnitude
information that was carrying what little signal remained, making it *worse*). **R2 fails**
because latent novelty barely separates the distributions (3.9 vs 5.1–5.9, heavily
overlapping), so a gate calibrated to be inert in-distribution stays nearly inert OOD.

The honest conclusion: the OOD failure is **not an input-scaling problem**. Feeding a
model-internal uncertainty signal through a predictor fitted on one difficulty distribution
does not transfer, because the signal's *rank-informativeness*, not merely its scale, is
distribution-specific.

## 8.3 H6: is simulation metacognition transferable? — SUPPORTED

Pre-registered in [PREREGISTRATION_H6.md](PREREGISTRATION_H6.md); report:
[results/REPORT_H6.md](results/REPORT_H6.md). `g_phi` is refitted on combinations of configs;
**no world model is retrained**. Each OOD config is partitioned 60/20/20 by episode id and the
20% test portion is evaluation-only in *every* regime, so "seen" and "unseen" cells are
comparable. **Bold = that config was never in training.**

### Predicted `H*` (actual: A 15.60, B 8.08, C 3.95, D 4.89)

| regime | test A | test B | test C | test D |
|---|---|---|---|---|
| A (IID only) | 15.68 | **21.60** | **13.83** | **21.21** |
| A+B | 15.46 | 7.82 | **12.08** | **5.07** |
| A+C | 14.76 | **9.88** | 6.10 | **8.64** |
| A+D | 15.59 | **6.89** | **12.01** | 4.54 |
| A+B+C | 14.59 | 7.73 | 6.25 | **5.23** |

### Spearman

| regime | test A | test B | test C | test D |
|---|---|---|---|---|
| A | +0.580 | **+0.218** | **+0.213** | **+0.135** |
| A+B | +0.584 | +0.755 | **+0.306** | **+0.771** |
| A+C | +0.576 | **+0.461** | +0.479 | **+0.510** |
| A+D | +0.580 | **+0.699** | **+0.311** | +0.847 |
| A+B+C | +0.577 | +0.754 | +0.477 | **+0.766** |

**Verdict: SUPPORTED**, on all three seeds, and **the size-matched control reproduces it**
(every regime subsampled to the A-only anchor count of 22,200; `A+B+C→D` MAE 3.62, ρ +0.762).
The gain is therefore attributable to *diversity of failure modes*, not to more training data.
In-distribution performance is preserved in every regime (A-test MAE moves by ≤ 0.17).

The critical cell is decisive: on **D, never seen in training**, `A+B+C` moves MAE from
**16.77 → 3.58** and Spearman from **+0.135 → +0.766**, and the prediction becomes properly
conservative (21.2 → 5.2 against a true 4.89). The anti-conservative failure of §8 is gone.

### The result is not a diversity ladder — one shift is enough

`A+B → D` (ρ +0.771) is **as good as** `A+B+C → D` (ρ +0.766). Adding a second shift buys
nothing. The discontinuity is between **zero** shifts and **one**, and it saturates
immediately. Whatever `g_phi` learns from a single shift is not shift-specific detail but
something closer to "do not trust the in-distribution uncertainty cue blindly".

### Metacognitive transferability is organised by failure-mode family

The exception is systematic, and it is the most interesting part:

| Held-out cell | Spearman | Restored (main / size-matched) |
|---|---|---|
| A+B → D | +0.771 | 3/3 · 3/3 |
| A+D → B | +0.699 | 3/3 · 3/3 |
| A+C → B | +0.461 | 3/3 · 3/3 |
| A+C → D | +0.510 | 3/3 · 3/3 |
| **A+B → C** | **+0.306** | **2/3 · 1/3** |
| **A+D → C** | **+0.311** | **1/3 · 1/3** |

**B and D transfer to each other almost perfectly; neither transfers to C.** B (unseen
layouts) and D (long corridors) are both *epistemic* shifts — the agent's uncertainty is about
geometry it has not seen. C is *aleatoric* — a 15% action slip, irreducible noise. Only
training on C fixes C (+0.479).

This is the **metacognitive transferability** outcome: simulation metacognition transfers
across shifts that share a failure *mechanism*, and fails across qualitatively different ones.

### Why — the rank-stability table explains it mechanically

| feature | ρ_A | ρ_B | ρ_C | ρ_D | RS = Var_D[ρ] |
|---|---|---|---|---|---|
| `imagined_disp_spread` | **−0.455** | −0.096 | −0.179 | −0.096 | 0.0227 |
| `free_ahead` | **+0.004** | **+0.586** | +0.024 | **+0.566** | **0.0791** |
| `ensemble_disagreement_mean` | −0.204 | −0.194 | −0.090 | −0.039 | 0.0054 |
| `latent_prior_std` | +0.029 | −0.009 | +0.053 | −0.057 | 0.0020 |

The two rows at the top are the whole story:

- `imagined_disp_spread` is the **strongest in-distribution** signal (−0.455) and collapses OOD
  (|ρ| ≈ 0.10–0.18). `g_A` leans on it, which is exactly why it fails out of distribution.
- `free_ahead` — how far the agent can currently *see* clear space ahead — is **useless
  in-distribution** (+0.004) but the **strongest OOD signal by far** (+0.586 on B, +0.566 on D).
  A predictor trained only on A has no reason to use it.

Shift exposure teaches `g_phi` to switch from the model-internal uncertainty cue to the
observation-geometry cue. And this closes the loop on C: `free_ahead` has ρ_C = **+0.024** —
useless on C too, because C's failures are noise-driven rather than geometry-driven. **The
feature that rescues B and D cannot rescue C**, which is precisely why the epistemic family
transfers internally and not to the aleatoric one.

Note also that no feature is both stable *and* useful: `latent_prior_std` has the lowest
instability (RS 0.0020) but ρ ≈ 0 everywhere. There is no distribution-invariant uncertainty
scalar here — which is why a *learned* predictor with shift exposure is needed rather than a
better hand-chosen signal.

## 8.4 H7: is this a property of world models, or of one implementation?

Three architectures trained on config A with identical data, seeds, targets and downstream
code — only the dynamics differ. **Single seed, K=3: an existence check, not a precision
estimate.** Report: [results/cross_arch/REPORT_H7.md](results/cross_arch/REPORT_H7.md).

| | dynamics | params |
|---|---|---|
| `gru_large` | GRU-RSSM, deter 512 / stoch 48 | 2.85 M |
| `gru_small` | GRU-RSSM, deter 256 / stoch 16 | ~0.8 M |
| `transformer` | causal attention, **no recurrence, no autoregressive latent feedback** | 2.87 M |

The `no_latent` feature set has identical *names* in all three (74 columns, asserted at
runtime), which is what makes the cross-architecture cells meaningful.

### H2 replicates in every architecture

| arch | `H*` on A | A MAE | A ρ | beats its own Claim-A ceiling |
|---|---|---|---|---|
| gru_large | 15.59 | 6.408 | +0.584 | **yes** (+1.355) |
| gru_small | 13.65 | 6.351 | +0.551 | **yes** (+1.050) |
| transformer | 16.58 | 6.432 | +0.513 | **yes** (+0.993) |

Each is compared against *its own* optimistically-fitted ceiling on *its own* `H*`. The core
positive result is **not an artefact of the GRU-RSSM implementation**: a model with no
recurrence and no latent feedback reproduces it at similar strength.

### The metacognitive signal is partially, not fully, architecture-general

`g_φ` fitted on one architecture's features, applied to another's A-test (Spearman):

| fitted on \ evaluated on | gru_large | gru_small | transformer |
|---|---|---|---|
| **gru_large** | +0.584 | +0.391 | +0.207 |
| **gru_small** | +0.356 | +0.551 | +0.288 |
| **transformer** | +0.363 | +0.362 | +0.513 |

A clear hierarchy: **within-architecture (0.51–0.58) > within-family (0.36–0.39) >
cross-family (0.21–0.36)**. Cross-architecture transfer is clearly above zero — there is a
shared, architecture-independent component to the uncertainty→horizon relationship — but it
loses roughly a third to two-thirds of the within-architecture signal. A horizon predictor is
therefore *partly* portable across world models and cannot simply be reused as-is.

### H6 does NOT replicate in the Transformer — and the reason is informative

| arch | ρ on held-out D, train A | train A+B | `H*` on D | zero-fraction on D |
|---|---|---|---|---|
| gru_large | +0.201 | **+0.724** | 5.04 | 0.293 |
| gru_small | −0.003 | **+0.745** | 5.13 | 0.295 |
| transformer | +0.257 | **+0.290** | **8.55** | **0.038** |

Shift exposure transforms both GRUs and barely moves the Transformer. The explanation is in
the last two columns: **the Transformer is a much better world model out of distribution.**
On D its imagination stays exact for 8.55 steps versus ~5.1 for the GRUs, and only **3.8%**
of its anchors collapse to `H* = 0` versus **~29%** for both GRUs.

For the GRUs, config D is largely a catastrophe, and what shift exposure teaches is mostly
"out of distribution, expect near-zero" — a large, easily-learned correction. The Transformer
has no such collapse to correct, so there is far less headroom for diversity to recover.

This sharpens H6 rather than contradicting it: **the size of the diversity effect is a
function of how badly the world model degrades under shift, which is architecture-dependent.**
The claim "diverse exposure restores transfer" should be stated as "restores transfer *to the
extent that the underlying model has degraded*", and a stronger world model needs the
correction less.

### Caveats

Single seed and K=3 (vs 3 seeds, K=5 in the main run), so no seed variance is available and
the ensemble-disagreement features are computed over fewer members. The Transformer was
trained for the same 6000 steps as the GRUs — equal gradient updates, not equal wall-clock
(it is ~5x slower per step). A fairer compute-matched comparison would give it fewer steps,
which would only weaken it, so its success here is not a compute artefact.

## 8.5 H8: does the failure-mode taxonomy *predict*? — epistemic yes, aleatoric untested

Six predictions frozen in [PREREGISTRATION_H8.md](PREREGISTRATION_H8.md) before E/F existed.
**E** = `SimpleCrossingS11N5` (new epistemic), **F** = config A + 15% observation-cell
corruption (a *different* aleatoric mechanism from C's action slip). Existing main-run world
models, no retraining.

### Spearman by regime (mean over 3 seeds; **bold** = config unseen in training)

| regime | test A | test B | test C | test D | test E | test F |
|---|---|---|---|---|---|---|
| A | +0.580 | **+0.218** | **+0.213** | **+0.135** | **+0.116** | **+0.472** |
| A+B | +0.584 | +0.755 | **+0.306** | **+0.771** | **+0.855** | **+0.373** |
| A+C | +0.576 | **+0.461** | +0.479 | **+0.510** | **+0.555** | **+0.417** |
| A+E | +0.581 | **+0.732** | **+0.313** | **+0.800** | +0.879 | **+0.423** |
| A+F | +0.587 | **+0.266** | **+0.231** | **+0.102** | **+0.110** | +0.569 |

| # | cell | threshold | ρ | seeds | holds |
|---|---|---|---|---|---|
| **P1** | A+B → E | ρ > 0.50 | **+0.855** | 3/3 | **YES** |
| P2 | A+B → F | ρ < 0.35 | +0.373 | 1/3 | no |
| **P3** | A+C → F | ρ > 0.50 | +0.417 | 0/3 | **no** |
| **P4** | A+E → D | ρ > 0.50 | **+0.800** | 3/3 | **YES** |
| P5 | A+F → C | ρ > 0.50 | +0.231 | 0/3 | no |
| **P6** | `free_ahead` informative on E, not F | — | ρ_E +0.533, ρ_F +0.019 | — | **YES** |

**Frozen verdict: NOT SUPPORTED.** (Noting a gap in our own rule: the pre-registration
assigned NOT SUPPORTED only when P1 fails, and PARTIAL when P1+P2 hold but P3 fails. P1
holding while P2 fails was unspecified; the scoring code defaulted to NOT SUPPORTED and we
report that rather than re-deriving a friendlier label after the fact.)

### The epistemic half is confirmed, and predictively

E was **never seen** in training, is a different maze family from B and D, and is a *hard*
config (`H*` 2.18, 42% of anchors at zero). It still behaves exactly as the taxonomy
predicted, in both directions:

- as a **recipient**: `A+B → E` = **+0.855** against a +0.116 IID-only baseline (P1);
- as a **donor**: `A+E → D` = **+0.800** and `A+E → B` = +0.732 — E teaches what B and D teach;
- the **mechanism** replicates on new data (P6): `free_ahead` is +0.533 on E, matching B
  (+0.586) and D (+0.566), while remaining useless in-distribution (+0.004).

An advance prediction about an unseen environment, confirmed at ρ +0.855 on 3/3 seeds, is the
strongest evidence in this project that the epistemic grouping is real rather than a
description fitted to B and D.

### The aleatoric half failed as a *manipulation*, not as a hypothesis

Config F is barely a shift. Its `H*` is **15.23** against A's 15.52, and the IID-only
predictor already reaches **ρ +0.472** on it — more than double its transfer to any other
config (+0.116 to +0.218). Observation noise leaves the action-conditioned `moved`/`dir`
predictions largely intact, so F sits near the training distribution.

That makes P3 and P5 uninformative about the taxonomy:

- **P3** (`A+C → F` > 0.5) asks whether an aleatoric donor helps on F, but there is little to
  help *with*; and every shift-exposed regime **underperforms the IID-only baseline on F**
  (0.373 / 0.417 / 0.423 vs **0.472**).
- **P5** (`A+F → C` > 0.5) asks whether F is a useful donor, but a config that is nearly
  in-distribution teaches nothing: the `A+F` row is essentially the `A` row (→B +0.266 vs
  +0.218, →D +0.102 vs +0.135).

We flagged F's `H*` ≈ A's *before* scoring the predictions, and it is measurable without
seeing any outcome, so this is not a post-hoc rescue. But the honest statement is that **the
aleatoric half of the taxonomy remains untested**, not that it is refuted. C is still its
only member.

### A new finding: shift exposure is not free

The F column exposes a cost that H6 could not see. On a **near-in-distribution** config,
shift-exposed predictors are *worse* than the IID-only one (0.373–0.423 vs 0.472), while
losing nothing on A itself (0.584 vs 0.580). Training on a distant shift pulls the predictor
away from the regime where the model is still reliable.

So H6's "one shift is enough" should be qualified: one shift buys a large gain on badly
degraded configs and costs a modest amount on mildly shifted ones. That trade is invisible
unless the evaluation suite spans a *range* of shift severities — which, before H8, ours
did not.

## 9. Novelty assessment

Against the [Phase 0 audit](docs/phase0_literature.md):

- **Not novel:** uncertainty correlates with rollout error (H1). Well established.
- **Moderately novel:** a *learned, per-state* horizon predictor that beats the best monotone
  function of any single uncertainty scalar, evaluated as a prediction task with an
  optimistically-fitted control. Existing a-priori methods
  ([Certified World Models](https://arxiv.org/abs/2606.13092),
  [Conformal Trust Horizons](https://arxiv.org/abs/2606.24946)) are analytic and explicitly
  orbit-wide/worst-case; the former states it does not predict which state will breach
  tolerance. We do exactly that, and beat online-truncation baselines by 1.96 MAE.
- **Novel and negative:** the horizon signal requires inspecting the imagined rollout;
  strictly a-priori features are insufficient. This narrows the gap to
  [Round-Trip Consistency](https://arxiv.org/abs/2608.00675) — that work also needs a rollout,
  though a more expensive bidirectional one — and refutes the strongest reading of the
  original hypothesis.
- **Novel and negative:** systematic *anti-conservative* failure under shift, empirically
  confirming the mechanism [Biased Dreams](https://arxiv.org/abs/2604.25416) predicted, now
  measured as a horizon-prediction error rather than an uncertainty-tracking error.

- **Novel and positive (H6, §8.3):** the OOD failure of uncertainty-based horizon prediction
  is **not intrinsic** — it is a training-diversity artefact, repaired by exposure to a single
  shift and surviving a size-matched control. The accompanying finding that transfer is
  organised by **failure-mode family**, with a mechanistic account (the predictor must switch
  from a model-internal cue to an observation-geometry cue, and no distribution-invariant
  uncertainty scalar exists in our feature set), is the strongest contribution here. We have
  not found this in the surveyed literature, which studies uncertainty transfer as a
  calibration problem rather than a rank-informativeness one.

Honest rating: **moderately novel overall**, with H6 the part most plausibly reaching
"strongly novel". The H2 positive is a sharper version of a known idea; the H3/H4/H5
negatives are useful but local; H6 is the result that generalises, and it is also the one
resting on the fewest configurations (see §11).

## 10. ICML assessment

| Axis | Rating | Reasoning |
|---|---|---|
| Novelty | **Medium** | Crowded neighbourhood; the per-state learned-predictor framing and the ceiling control are the new parts. |
| Significance | **Medium** | The OOD failure is directly relevant to anyone using uncertainty to gate imagination. |
| Technical quality | **Medium-high** | Pre-registered, 3 seeds, clustered bootstrap CIs, executable leakage test, optimistic-ceiling control, privileged bound. |
| Empirical strength | **Medium-low** | One environment family (MiniGrid), one architecture, 2.8 M-parameter model, no pixels, H4 unresolved. |
| Reproducibility | **High** | Frozen config/seeds/splits, provenance per run, deviations logged, one-command reproduction. |

**Likely reviewer objections, in the order I expect them:**

1. *"MiniGrid only, and a symbolic-state metric."* Fair and serious. The mechanism (visible
   horizon vs planned path) may not transfer to pixels or continuous control.
2. *"21 % of targets are right-censored at `H_max = 30`."* Real; MAE is inflated and the
   upper tail is unidentified. A larger `H_max` would settle it.
3. *"The win over the ceiling is ~1 step of MAE on a target with sd 10."* Statistically solid
   but modest in absolute terms; R² = 0.33 means most variance is unexplained.
4. *"If it needs the imagined rollout, how is this different from truncation?"* Answered: we
   beat online truncation by 1.96 MAE, but the reviewer is right that the a-priori framing
   does not survive.
5. *"The OOD failure undermines the practical claim."* Agreed, and we say so.

## 11. Next experiment (single most valuable)

Both previously-recommended experiments have now been run: H5 refuted the input-scaling
explanation (§8.2) and H6 established that transfer *is* achievable but is organised by
failure-mode family (§8.3).

The taxonomy — *epistemic* shifts (B, D) transfer to each other, the *aleatoric* shift (C)
neither donates nor receives — is currently inferred from **three** configs, one of which is
the sole member of its family. That is a hypothesis fitted to a 2-vs-1 split, and it is the
weakest load-bearing claim in the project.

**[RUN — see §8.5]** Proposed experiment — a falsifiable test of the failure-mode-family taxonomy. Add two new
configs chosen to make an advance prediction that can fail:

- **E, epistemic**: a new layout topology (e.g. `SimpleCrossingS11N5`) — a geometric shift
  with no added noise.
- **F, aleatoric**: config A with **observation noise** (random symbol corruption in the
  egocentric view) rather than action slip — a *different* aleatoric mechanism from C.

Registered predictions, made before running: `A+B → E` transfers (ρ > 0.5) while `A+B → F`
does not (ρ < 0.35); and `A+C → F` transfers despite C and F having different noise sources,
because both are aleatoric. If instead `A+C → F` fails, the organising principle is not
"epistemic vs aleatoric" but something narrower — most likely *which feature carries the
signal* — and the taxonomy claim must be withdrawn.

This is the right next step because it converts the most interesting result in the project
from a post-hoc description of three configs into a **predictive** claim. It requires
collecting two new configs and one feature-extraction pass per seed (no world-model
retraining, since the model transfers by construction on egocentric observations), so it is
roughly one Kaggle GPU session.
