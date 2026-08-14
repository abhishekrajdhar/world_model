# Learning to Monitor the Reliability Horizon of Imagined Futures

*Polished manuscript (markdown mirror of `paper/main.tex`). The LaTeX version is
authoritative; all numbers there are emitted programmatically from frozen artifacts. The
original draft is preserved unchanged in `PAPER.md`.*

---

## Abstract

An agent that plans by imagining must decide how far to trust what it imagines. We study
whether a world model can learn a **monitor** of that trust: a predictor of the *reliability
horizon* `H*`, the number of consecutive imagined steps whose predicted state matches the
environment exactly. In partially observed gridworlds, a learned monitor predicts `H*` with
Spearman ρ = 0.573 ± 0.011 and improves on an *optimistically fitted* monotone single-scalar
ceiling by 1.04 MAE on average, with per-seed bootstrap confidence intervals excluding zero.
The monitor requires information from the imagined rollout: restricted to time-`t` features it
fails to beat that ceiling on any seed. Under distribution shift the monitor fails
**anti-conservatively**, predicting horizons near 20 where the true horizon is 4–8; it is most
confident where the world model is least reliable. Two repairs we tested — rank-normalising
the monitor's inputs and gating on latent novelty — do not recover it: the uncertainty signals
retain aggregate separation between distributions while losing within-distribution
rank-informativeness. Refitting the monitor with a single shifted configuration in its training
set moves a never-seen configuration from MAE 16.77 / ρ 0.135 to MAE 3.58 / ρ 0.766, and a
size-matched control indicates this is not explained by sample count alone. The restoration is
selective: shifts that alter geometry transfer strongly to other geometric shifts (ρ 0.73–0.86)
but weakly to noise-driven ones (ρ ≈ 0.31). The core result also appears in a world model with
no recurrence and no autoregressive latent feedback, though monitors transfer only partially
across architectures. These results suggest that simulation-based reliability monitoring is
measurable and not architecture-specific, but that its calibration is learned from the
distribution of failures the world model has experienced rather than being an invariant
property of uncertainty.

---

## 1. Introduction

World models are used to plan by imagining. The question usually asked of such a model is
whether it predicts the future accurately. The question an acting agent needs answered is
sharper: *for how many steps does this particular imagined rollout remain worth acting on?*

The standard answer thresholds an uncertainty signal — roll forward, stop when disagreement or
entropy crosses a bound. That rests on a premise prior work supports: uncertainty correlates
with realised rollout error (**Claim A**). It licenses ranking states by riskiness, nothing
more.

We study **Claim B**: an agent can learn a *monitor* predicting the horizon itself, per state,
as a quantity. The distinction matters because a threshold rule is only as good as the monotone
map from uncertainty to trust. If that map exhausts the signal, nothing beyond Claim A has been
shown.

Our central device is therefore a control, not a baseline: for each uncertainty scalar we fit
an isotonic regression to `H*` **on the evaluation data itself** — the best attainable
performance of *any* monotone function of that signal, with no generalisation gap. Beating it
is the minimum evidence that Claim B says something new.

**Contributions.** (1) An operational reliability horizon with an exactly checkable target, and
evidence that a learned rollout monitor predicts it beyond an optimistic single-scalar ceiling.
(2) Evidence that imagined-rollout information is necessary. (3) An anti-conservative failure
under shift, traced to loss of rank-informativeness rather than of scale. (4) Evidence that
shift exposure can substantially restore transfer, is not explained by sample count alone, and
is selective across shift types and architectures.

## 2. Related Work

**World models.** Latent dynamics models underpin modern model-based control (Ha & Schmidhuber,
2018; Hafner et al., 2019; 2020; 2021; 2023). We study the horizon over which imagination is
exact rather than control performance.

**Uncertainty in learned dynamics.** Deep ensembles and their disagreement are the standard
epistemic signal (Lakshminarayanan et al., 2017; Chua et al., 2018); Li et al. (2025) propagate
epistemic uncertainty through long autoregressive rollouts to make offline model-based RL work
on real robots. These establish Claim A; we treat the best monotone map from such a scalar as a
control to beat.

**Adaptive rollout length.** Model-based RL truncates rollouts at an uncertainty threshold
(Janner et al., 2019), and rollout length can be tuned online as a meta-level decision problem
(Bhatia et al., 2022). ELVIS (Du et al., 2026) gates a time-varying λ via a UCB score from an
ensemble of latent critics — this adapts the *value estimator*, not a predicted horizon.

**Horizon certificates.** Wang (2026a) derives predictability certificates for equivariant
latent world models from the finite-time Lyapunov spectrum. Wang (2026b) calibrates
trust-horizon curves by split conformal prediction and transports them across a symmetry orbit,
reporting a median certified-to-measured ratio of 0.67 and **zero anti-conservative violations**
across stable audits. Both require known group symmetry and are orbit-level and worst-case.
Ours is a learned per-state estimate with no symmetry assumption and no conservatism guarantee
— the anti-conservative failure in §6.3 is precisely what a conformal construction excludes.

**Self-estimated rollout error.** Scheinker (2026) uses forward-then-backward round-trip
discrepancy in a bidirectional latent diffusion model as a self-supervised error proxy
(Spearman 0.91–0.98) on magnetohydrodynamics, mixing layers and face video. It needs a model
trained to run in both directions and an extra backward pass, and has no acting agent; our §6.2
result puts us on the same side of the a-priori boundary at lower cost. Song & Cai (2026)
explain rollout error post hoc in graph world models via a topology factor (spectral radius)
and a model factor — model-level, not per-state.

**Failure prediction and data coverage.** Berger et al. (2026) report that in RSSM latent
rollouts ensemble disagreement captures local epistemic uncertainty but does not reflect error
accumulated over long rollouts, because rollouts are drawn toward attractor regions where
uncertainty declines. Hansen & Wang (2026) argue hallucination in large visual world models is
predictable from coverage signals and preventable by coverage-aware collection — "inherently a
data coverage issue". Our §6.5 result is convergent evidence from a different direction.

**Calibration and shift.** Calibration (Guo et al., 2017), selective prediction (Geifman &
El-Yaniv, 2017) and conformal methods (Angelopoulos & Bates, 2021) target when a prediction
should be trusted; our quantity is a *depth* rather than a label and the failure is a loss of
ranking rather than of coverage. Degradation of uncertainty under shift is documented (Ovadia
et al., 2019; Hendrycks & Gimpel, 2017); we add that here it inverts the safety-relevant
direction.

## 3. Problem Formulation

Given an action sequence known at planning time, the model produces an imagined rollout
`τ̂_{t:t+H}`. With `E_H = d(ŝ_{t+H}, s_{t+H})` and tolerance ε:

```
H*(s_t) = max { H ≤ H_max : E_h < ε  for all h ∈ 1..H }
```

a **first-failure** definition — once imagination diverges it is operationally unusable even if
it later re-synchronises. `H_max = 30`.

**Instantiation.** Observations are 5×5 egocentric symbolic views. The model predicts
size-agnostic state *factors* (direction; `moved`, whether the transition changed the agent's
cell; carrying; door state) and position is reconstructed by integrating `moved` along `dir`.
`E_H` is the Manhattan error of that integration and ε = 1, so `E_H < ε` means the model places
the agent in exactly the right cell. The integration identity is verified exactly on the true
factors for every configuration before modelling. The load-bearing factor is `moved` — whether
the cell ahead is free — which is precisely what partial observability hides.

**The monitor.** `ĝ_φ(x_t)` outputs a distribution over `H* ∈ {0..30}`, giving a point estimate
and `P(H* ≥ h)`. Features group into *a-priori* (belief state, one-step uncertainty, current
observation statistics), *rollout* (per-step disagreement, entropy, spread of imagined
displacement), and *actions* (the planned sequence). `x_t` may not depend on any observation,
state or error at time > t; this is verified by corrupting every future field and checking all
features are bit-identical while `H*` changes.

**The Claim-A ceiling.** For scalar `u`, `C_A(u) = min_{m monotone} (1/N) Σ |m(u_i) − H*_i|`,
computed by isotonic regression fitted on the evaluation split. It has no generalisation gap
and is optimistic by construction.

## 4. World Models

Three world models share data, seeds, targets and all downstream code, differing only in
dynamics. **M1/M2**: GRU-based recurrent state-space models (deter 512/stoch 48; deter
256/stoch 16). **M3**: causal self-attention, **no recurrence and no autoregressive latent
feedback** — observations are replaced by a learned mask token where unavailable, so
imagination is a single masked forward pass. M3 exposes the same interface and head set, so
`H*`, the features and every downstream computation are identical across architectures.

## 5. Experimental Setup

A (`DoorKey-16x16`) is the training distribution; shifts are B (`FourRooms`), C (A + 15% action
slip), D (`MultiRoom-N6`), E (`SimpleCrossingS11N5`), F2 (observation noise), F3 (20% action
dropout). All decision rules were frozen before the corresponding evaluation across five
pre-registration documents. Layout seeds are disjoint across world-model training, monitor
training, validation and test, and the monitor trains on episodes the world model never saw.
Three seeds, K = 5 members, episode-clustered bootstrap over 2000 resamples.

**Two measurement details.** 21% of A-test targets sit at the ceiling `H_max = 30` and are
right-censored, so MAE is inflated and the upper tail unidentified. And the main run evaluates
shifts on the *full* split while the transfer experiments partition each shifted configuration
60/20/20 and evaluate on the held-out 20% — baselines therefore differ slightly between
sections, and numbers should be compared within a table, not across.

Of nine pre-registered hypotheses, three were supported and five were not; all are reported in
the main text.

## 6. Results

### 6.1 The monitor predicts the horizon

On A-test `H*` has mean 15.60 and IQR 20–21. The monitor attains MAE 6.416 ± 0.087 and
ρ = 0.573 ± 0.011 (the variant selected on validation differed by seed, so fixed variants are
also reported). Against the optimistic ceiling it improves by 1.080 / 1.102 / 0.936 MAE, with
paired bootstrap lower bounds 0.978 / 1.010 / 0.854 — all excluding zero. Against a constant
predictor, by 2.51–2.63 MAE. In-distribution calibration is good (ECE 0.023–0.041).

Two comparisons sharpen this. The monitor beats **online truncation** by 1.99 MAE. And a
**privileged** predictor given the true local 11×11 grid plus the ordered action sequence does
*worse* (ρ 0.226). Knowing the environment is not sufficient; what the monitor exploits is the
model's relationship to it.

### 6.2 Rollout information is necessary

Restricted to strictly time-`t` features, the monitor reaches MAE 8.065 ± 0.087 — *worse* than
the ceiling by 0.464 / 0.630 / 0.735. The signal lives in the imagined trajectory's internal
disagreement. The honest reading of "before observing the future" is: before observing it, but
after imagining it.

### 6.3 Under shift the monitor fails anti-conservatively

| configuration | true `H*` | predicted | ρ |
|---|---|---|---|
| A (in distribution) | 15.60 | 15.60 | +0.573 |
| B (unseen layouts) | 7.75 | **21.01** | +0.219 |
| C (action slip) | 4.11 | **12.82** | +0.271 |
| D (long corridors) | 5.11 | **19.97** | +0.165 |

A monitor that degraded to noise would be useless; one that becomes *more* confident where the
model is least reliable is worse than useless for gating imagination.

### 6.4 Simple repairs do not recover it

Rank-normalising the uncertainty features, and a novelty gate calibrated to be inert in
distribution, both fail on all three seeds. Predictions remain near 20 against true horizons of
4–8, and rank-normalisation makes C worse. §7 explains why.

### 6.5 Exposure to a shifted distribution restores transfer

Refitting the monitor on combinations of configurations (no world model retrained; each shifted
configuration partitioned 60/20/20 with the test portion evaluation-only in every regime):

| training regime | test A | test B | test C | test D |
|---|---|---|---|---|
| A | 0.580 | 0.218 | 0.213 | 0.135 |
| A+B | 0.584 | *0.755* | 0.306 | **0.771** |
| A+C | 0.576 | 0.461 | *0.479* | 0.510 |
| A+B+C | 0.577 | *0.754* | *0.477* | **0.766** |

On D, never seen, `A+B+C` moves MAE 16.768 → 3.584 and ρ 0.135 → 0.766. A size-matched control
reproduces it (MAE 3.622, ρ 0.762), so the effect is **not explained by sample count alone**;
we do not claim diversity is its only cause. Adding a second shift changes little
(`A+B → D` = 0.771).

### 6.6 Architecture generality and monitor portability

All three architectures beat their own ceiling: +1.355 / +1.050 / +0.993 MAE, lower bounds
1.251 / 0.945 / 0.901. The phenomenon is not an artefact of recurrent latent feedback — it
appears in a model with none. **These runs use a single seed with K = 3**; they establish
existence, not a precision estimate.

Applying a monitor across architectures gives ρ 0.513–0.584 within an architecture, 0.356–0.391
within the GRU family, and 0.207–0.362 across families. Transfer is above zero but a monitor
cannot be reused as-is.

The restoration effect **does not replicate in the Transformer** (0.257 → 0.290). It is a
considerably better world model under shift (`H*` 8.55 on D vs ≈5.1; 3.8% of anchors at zero vs
≈29%), so much of what exposure teaches the GRUs — "under shift, expect near-zero" — has less
to correct. This is consistent with the effect scaling with how badly the underlying model
degrades, though we test a single contrast.

### 6.7 Restoration is selective across shift types

We pre-registered a two-family account and tested it predictively. The geometric half was
confirmed: on E, never seen and from a different maze family, `A+B → E` reaches ρ 0.855 against
a baseline of 0.116, and E acts as a donor (`A+E → D` = 0.800). The noise half was not:
`A+C → F3` reaches 0.444 against a pre-registered bar of 0.50. **We therefore withdraw the
taxonomy** and report only the asymmetry:

| donor ↓ / recipient → | geometric | noise-driven |
|---|---|---|
| geometric (B, E) | 0.73–0.86 | ≈ 0.31 |
| noise-driven (C, F3) | 0.39–0.54 | 0.44–0.47 |

Two caveats: the pattern rests on four donors in one environment family; and donors differ in
*severity* as well as kind, so an account in which transfer depends on how much the donor
degrades the model is not excluded.

### 6.8 A weak test of downstream utility

Adaptive-horizon planning reached success 0.517 ± 0.065 against 0.933 ± 0.032 for the best fixed
horizon, at 5.49× the compute. But success fell *monotonically* with horizon, so the frontier
had no interior optimum and an adaptive rule had nothing to find; the detour manipulation did
not take (mean detour 0.13 cells). We report this as a **weak test**, not a refutation.

## 7. Mechanistic Analysis

Write `ρ_D(u)` for the within-distribution rank correlation between feature `u` and `H*` on
distribution `D`. A shift can do two different things:

- **scale shift**: `E_{D'}[u] ≠ E_D[u]` while `ρ_{D'}(u) ≈ ρ_D(u)`
- **rank shift**: `|ρ_{D'}(u)| ≪ |ρ_D(u)|`

A monotone reparameterisation — rank normalisation included — absorbs a scale shift. It cannot
absorb a rank shift, because it preserves the ordering it is given, and under a rank shift that
ordering has stopped carrying information.

The dominant in-distribution feature, spread of imagined displacement across ensemble members,
undergoes a rank shift. Its mean *rises* from 0.517 to 0.986 / 1.455 / 0.971 on B / C / D — so
aggregate separation is retained — while `ρ_D` collapses from −0.446 to −0.111 / −0.191 /
−0.089. The signal still reports *that* matters are harder; it stops reporting *which states*
will fail. That is the direct explanation of §6.4.

Two further observations. Predictive entropy moves the *wrong* way under shift (`ρ_D` −0.394 →
+0.001 on D) whereas ensemble spread retains its sign. Berger et al. (2026) report a related
but distinct effect — uncertainty *declining along* an extended rollout drawn into attractor
regions. We measure across-distribution means and within-distribution rank correlations, not
the within-rollout trajectory of uncertainty, so our observations neither confirm nor
contradict that account. What our data do show is that the failure is not confined to one
signal family, and that no measured feature is both stable and useful: the one with lowest rank
instability has ρ ≈ 0 everywhere.

This also explains §6.7. One feature — how far the agent can currently see unobstructed — is
*useless* in distribution (+0.004) and *dominant* under geometric shift (+0.586, +0.566, +0.533
on B, D, E). A monitor trained only on A has no reason to use it; geometric shift exposure
teaches it to. On noise-driven configurations it is inert (+0.024, −0.015, −0.031). A cue that
rescues geometric shifts cannot rescue noise-driven ones, and we find no equivalent carrier on
the noise side — which is why those donors transfer broadly but never strongly.

## 8. Limitations

**One environment family** — MiniGrid, symbolic observations, ~2.8M-parameter models. The
mechanism is instantiated by a domain-specific feature; whether an analogue exists for
pixel-based or continuous-control world models is untested. **Censoring** — 21% of A-test
targets sit at `H_max`, inflating MAE and leaving the upper tail unidentified. **Effect size** —
R² = 0.318; the margin over the optimistic ceiling is ~1 MAE against a target with s.d. ≈ 10.3:
robust, and quantitatively modest. **Architecture evidence is single-seed** (K = 3): it
establishes the phenomenon occurs without recurrence, not comparisons between architectures.
**Few shift families, and a severity confound** — the asymmetry rests on four donors; two
configurations failed a pre-registered validity gate because they barely degraded the model, so
the noise side has two instruments; donor severity and type covary. **Downstream utility
untested** — the planning experiment was a weak test.

## 9. Discussion

A world model can learn a useful monitor of the reliability horizon of its imagined futures,
but that monitor is distribution-dependent and only partially transferable across architectures
and shift types.

The practical reading is uncomfortable for uncertainty-gated imagination. A monitor fitted where
the model works will be most confident where the model fails, and the two obvious input-level
corrections do not help, because the problem is that features stop *ranking* states rather than
that they are mis-scaled. What does help is exposure to failures — selectively, and less so when
the world model is already strong under shift.

This suggests reading reliability monitoring less as reading out an intrinsic property of
uncertainty, and more as a learned model of one's own failures whose validity is inherited from
the distribution of failures encountered in training. The asymmetry sharpens it: what a monitor
learns from one kind of failure may be a specific cue rather than a general capacity for
caution.

## 10. Conclusion

We defined an operational reliability horizon for imagined rollouts and showed that a learned
monitor predicts it beyond an optimistically fitted monotone single-scalar ceiling, provided it
may inspect the imagined rollout. The monitor fails anti-conservatively under distribution
shift; rank normalisation and novelty gating do not repair it; exposure to an appropriate
shifted distribution substantially restores transfer without being explained by sample count
alone; and that restoration is selective across shift types and does not replicate in a world
model that degrades less. Of nine pre-registered hypotheses, three were supported and five were
not; the negative results carry most of the content.
