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

Overall, against the pre-registered decision rule in §11 of the protocol: **SUPPORTED**, with
the scope restricted to in-distribution states and to predictors allowed to inspect their own
imagined rollout.

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

Honest rating: **moderately novel**, not "potentially major". The positive result is a
sharper version of a known idea; the negative results are the contribution most likely to
matter.

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

**Test whether the OOD failure is repairable by making the horizon predictor's inputs
scale-free.** The diagnosis is that `g_φ` consumes uncertainty magnitudes calibrated to
config A's difficulty. The concrete experiment: train `g_φ` on **rank-normalised**
uncertainty features (per-episode or per-batch quantile transform of disagreement/entropy)
plus an explicit novelty score (latent density under the training distribution), then
re-evaluate on B/C/D **without refitting**. Success criterion, fixed in advance: mean `Ĥ*`
must fall on all three OOD configs and Spearman must exceed 0.3 on at least two.

This is the highest-value next step because the in-distribution result is already established
and the OOD failure is what blocks every practical use of a reliability horizon — an agent
that is confident precisely when it is wrong is worse than one with a fixed conservative
horizon.
