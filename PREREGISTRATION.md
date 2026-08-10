# Pre-Registration: Reliable Imagination Horizon (RIH)

**Status: FROZEN.** Written before any data was collected, any model was trained, and any
result was observed. Any deviation after this point must be recorded in
`DEVIATIONS.md` with a timestamp and justification.

Git commit at freeze time: recorded in `results/prereg_freeze.json`.

---

## 1. Research question

Can an agent estimate, from information available at time `t` only, how many steps of its
own imagined future will remain trustworthy — *before* observing that future?

## 2. Hypothesis (sharpened after Phase 0 literature audit)

The naive hypothesis ("uncertainty predicts error") is already established. The audit
(see `docs/phase0_literature.md`) shows the following are also already demonstrated:

- a-priori horizon bounds from **analytic** signals (Jacobian/Lyapunov spectrum, one-step
  residual + expansion estimate, conformal calibration) — but these are *orbit-wide and
  worst-case*, and explicitly **do not** predict which specific state will breach tolerance;
- **online truncation** of rollouts once uncertainty crosses a threshold (MBPO/MACURA-style)
  — this observes the rollout as it unfolds;
- **round-trip consistency**, which predicts rollout error but requires actually performing
  a forward+backward rollout (2x inference).

We therefore pre-register the following, stronger and narrower claims:

- **H1 (Claim A, weak / replication).** Uncertainty signals available at time `t` correlate
  with future imagination error `E_H`. *Expected to hold; not sufficient.*
- **H2 (Claim B, main).** A learned predictor `g_phi` using only time-`t` information
  predicts the per-state ground-truth horizon `H*` with accuracy **strictly better than the
  best monotone function of any single uncertainty scalar** (the Claim-A ceiling), and
  better than the constant/marginal predictor.
- **H3 (OOD).** `g_phi`'s predicted horizon **decreases before** catastrophic model error
  under distribution shift, i.e. it transfers to environments it was not fitted on.
- **H4 (Claim C, utility).** Adaptive-horizon planning using `Ĥ*` achieves reward >= the
  best fixed horizon at strictly lower imagination compute.

**H2 is the primary hypothesis.** H1 is a sanity precondition. H3 and H4 are secondary.

## 3. Environments

Base: MiniGrid, egocentric partial observation (`agent_view_size=5`, symbolic 5x5x3).
Because observations are egocentric, all configs share one observation space and grid size
is not a confound.

| Config | Description | Purpose |
|---|---|---|
| **A** | `DoorKey-8x8`, slip=0.0 | training distribution |
| **B** | `MultiRoom-N4-S5-v0`, slip=0.0 | unseen layout topology (structural shift) |
| **C** | `DoorKey-8x8`, slip=0.15 | modified/stochastic dynamics |
| **D** | `DoorKey-16x16`, slip=0.0 | long-horizon / larger layouts |

World model and `g_phi` are fitted **on config A only**. B, C, D are OOD evaluation.

`slip`: with probability `p` the executed action is replaced by a uniformly random action.

## 4. Splits and seeds

Layout seeds are disjoint by construction. The horizon predictor gets its **own** split,
held out from world-model training: if `g_phi` were fitted on episodes the world model was
trained on, `H*` there would be inflated by memorisation and unrepresentative of deployment.

- A-wmtrain: `[0, 3000)`      — world-model training only
- A-gtrain:  `[3000, 4200)`   — horizon-predictor training (world model never saw these)
- A-val:     `[4200, 4800)`   — thresholds, model selection, honest baseline calibration
- A-test:    `[4800, 5400)`   — **FINAL EVALUATION ONLY**
- B/C/D-test: `[100000, 100600)`

Belief context: `CONTEXT = 16` observed steps are filtered to form the belief at the anchor
(within the 5-20 range the world model is trained on). Anchors are subsampled with
stride 4 within episodes.

Model training seeds: `{0, 1, 2}` (3 independent repeats of the entire pipeline).
Ensemble member seeds within a repeat: `{0..K-1}`, `K=5`.

## 5. World model

Encoder (symbolic obs -> embedding) -> recurrent latent (GRU) with Gaussian prior/posterior
heads (RSSM-lite) -> decoder heads. Target size 5M-30M parameters. Mixed precision on GPU.

**Decoder heads predict size-agnostic state factors:**
- `dir_t` in {0,1,2,3} (categorical)
- `moved_t` in {0,1} — whether the transition into step `t` changed the agent's cell
- `carrying_t` in {0,1}, `door_open_t` in {0,1}

Absolute displacement is reconstructed by integration:
`pos_{t+H} - pos_t = sum_h moved_h * unit(dir_{h-1})`.
This makes position error a deterministic function of the predicted factors and keeps the
model independent of grid size. The scientifically load-bearing quantity is `moved`: whether
the cell ahead is free is exactly what partial observability hides.

## 6. Error metric and RIH definition — FROZEN

**Primary distance:**
`E_H = Manhattan(predicted_displacement_H, true_displacement_H)` in grid cells.

**Threshold:** `epsilon = 1`. Since displacement is integer-valued, `E_H < 1` means
`E_H = 0`, i.e. *the model still places the agent in exactly the right cell*.

**Reliable Imagination Horizon:**
```
H*(s_t) = max { H in [0, H_max] : E_h = 0 for all h in {1..H} }
```
(first-failure definition: the number of consecutive correct imagined steps).
`H_max = 30`. `H*` therefore takes values in `{0,...,30}`.

Rationale for first-failure rather than the non-monotone `max{H : E_H < eps}`: once
imagination diverges it is operationally unusable for planning even if it re-synchronises by
coincidence. **Both are computed**; the first-failure version is primary, the non-monotone
version is reported as a robustness check.

**Secondary error metrics** (reported, not used to define `H*`): full-state exact match,
direction error, decoder NLL, latent distance.

`epsilon = 1` is fixed a priori by the discreteness of the state space, not chosen from data.

## 7. Prediction settings

- **Setting II (primary, planning-relevant).** `g_phi(z_t, U_t, a_{t:t+H_max}) -> Ĥ*`.
  The candidate action sequence is genuinely available at planning time.
- **Setting I (robustness).** `g_phi(z_t, U_t) -> Ĥ*`, marginalising over action sequences
  drawn from the collection policy.

**Information the predictor MUST NOT see** (enforced by construction in
`src/horizon/features.py`, asserted in tests): any observation, state, reward, or error at
time > t; any quantity derived from the real future.

Permitted: current + past observations, current latent, planned actions, model uncertainty,
ensemble disagreement, hidden state.

## 8. Uncertainty signals (Phase 4)

- **A. Predictive entropy/variance** — entropy of decoder categorical heads under the prior.
- **B. Ensemble disagreement** — K=5 independently trained models; Jensen-Shannon divergence
  and variance across members' predicted head distributions.
- **C. Latent uncertainty** — (i) trace of prior covariance; (ii) prior/posterior KL at
  time `t` (the model's one-step latent surprise on *already observed* data).

## 9. Baselines (all fitted on A-train/A-val only)

1. **Fixed horizon** — constant = median `H*` on A-val (the marginal/null predictor).
2. **Random horizon** — uniform over the observed `H*` range.
3. **Current-step error** — 1-step prediction error at `t`, isotonic-calibrated to `H*`.
4. **Model confidence** — decoder entropy at `t`, isotonic-calibrated.
5. **Ensemble disagreement** — isotonic-calibrated.
6. **Latent distance** — distance to nearest training latent, isotonic-calibrated.
7. **Hand-crafted heuristic** — features computable without the world model
   (visible free cells ahead, wall distance in view, number of `forward` actions in the plan).
8. **Oracle** — uses the true future; upper bound only.
8b. **Privileged-state upper bound** — a predictor given the **true local grid** (an 11x11
   patch revealing cells the agent cannot see) plus the planned actions. This is *not* an
   agent-available method. It separates two very different explanations of residual error:
   *"the agent cannot know this"* (partial observability) versus *"g_phi is weak"*. In
   deterministic configs `H*` is a deterministic function of the true state and the action
   sequence, so this bound is the ceiling on what any predictor with full state access could
   achieve, while `g_phi` is limited to what the agent has actually observed.

**Primary variant selection.** `g_phi` is trained in several feature-group variants (full,
strict a-priori, uncertainty-only, latent-only, hand+actions, actions-only). The **primary**
variant is the one with the lowest **A-val** MAE. Test data is never used for this choice.
All variants are reported regardless.

**Claim-A ceiling (the critical control).** For each single uncertainty scalar `u`, fit an
isotonic regression `u -> H*` **on the test set itself** (deliberately optimistic). This is
the best possible performance of any monotone function of that scalar. H2 is supported only
if `g_phi` beats this optimistically-fitted ceiling.

## 10. Metrics

Predicted vs actual `H*`: MAE, RMSE, Spearman rho, Pearson r, R^2, and Kendall tau.
Calibration: reliability diagram for `P(H* >= h)`, expected calibration error (ECE).
All with 10,000-sample bootstrap 95% CIs over anchor states, clustered by episode.

## 11. Decision rules — FROZEN

Let `M` = MAE of `g_phi`, `M_const` = MAE of Baseline 1, `M_ceil` = MAE of the best
optimistically-fitted Claim-A ceiling. All on A-test, averaged over 3 training seeds.

- **SUPPORTED** if, on A-test: (a) `M < M_const` with non-overlapping bootstrap 95% CI,
  **and** (b) `M < M_ceil` with paired bootstrap 95% CI for `M_ceil - M` excluding 0,
  **and** (c) Spearman rho(Ĥ*, H*) > 0.5 with CI excluding 0.3.
- **PARTIALLY SUPPORTED** if (a) holds but (b) or (c) fails — i.e. the signal is real but
  not stronger than the known weak claim.
- **NOT SUPPORTED** if (a) fails.
- **INCONCLUSIVE** if `H*` has insufficient variance on A-test (defined as: interquartile
  range of `H*` < 2 steps, or > 80% of anchors at the `H_max` ceiling), because the
  prediction task would then be degenerate. This check is run on **A-val**, before test.

**Degeneracy check runs on validation first.** If A-val `H*` is degenerate by the above
criterion, the experiment is diagnosed and the environment/threshold is revised *before*
test data is touched, and that revision is logged in `DEVIATIONS.md`.

- **H3 SUPPORTED** if mean `Ĥ*` on B/C/D is significantly lower than on A-test AND the
  ranking metric (Spearman) remains > 0.3 on at least 2 of 3 OOD configs.
- **H4 SUPPORTED** if adaptive planning reward >= best fixed-horizon reward (paired over
  identical env seeds, CI excluding a 5% relative drop) AND mean imagined states per
  decision is strictly lower with non-overlapping CI.

## 12. Compute plan

Pilot (tiny: 200 episodes, K=2, 2 epochs) validates the pipeline end-to-end and the metric
code before any full run. Full run only after pilot passes assertion suite.

## 13. What would falsify the hypothesis

- `g_phi` fails to beat the constant predictor -> internal state carries no horizon information.
- `g_phi` matches but does not beat the isotonic single-scalar ceiling -> the result reduces
  to the already-known weak claim (Claim A), and we report that.
- `H*` is degenerate (all at ceiling or all at 0) -> the environment cannot test the question.

A negative result is a valid outcome and will be reported as such.
