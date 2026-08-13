# Learning to Monitor the Reliability Horizon of Imagined Futures
### When it works, when it fails, and what experience restores it

---

## Abstract

A model-based agent that plans by imagining must decide how far to trust what it imagines. We
study whether an agent can learn a **monitor** of that trust: a predictor of the **Reliable
Imagination Horizon** `H*(s_t)`, the number of consecutive imagined steps whose predicted state
matches the environment exactly, estimated from information available at time `t`.

Our claim is deliberately bounded: **world models can learn a useful monitor of the reliability
horizon of their imagined futures, but that monitor is distribution-dependent and only
partially architecture-transferable.**

The evidence follows one arc:

**Monitoring works.** A learned monitor predicts `H*` with Spearman **0.573 ± 0.005** and beats
the best *monotone function of any single uncertainty scalar* — fitted optimistically **on the
test set** — by **1.02 MAE** (95% CI excludes zero on all three seeds). This exceeds the weak
claim, already established in prior work, that uncertainty correlates with rollout error. The
monitor must be allowed to inspect its own imagined rollout: restricted to strictly time-`t`
features it fails to beat that control.

**Distribution shift breaks it, anti-conservatively.** Out of distribution the monitor predicts
*longer* horizons (≈20) where the true horizon collapses (4–8). It is most confident precisely
where the world model is worst — a failure mode more damaging than ordinary calibration decay.

**Simple repairs fail.** Rank-normalising the monitor's inputs and gating on latent novelty both
fail on every seed. The diagnostic shows why: the uncertainty signals retain *global*
separation between distributions but lose *per-state rank-informativeness*, with the dominant
signal's within-split correlation collapsing from −0.455 to −0.09…−0.20.

**Exposure to a shifted distribution can restore transfer.** Refitting the monitor with one
shifted configuration in its training set moves a never-seen configuration from MAE 16.77 /
ρ 0.135 to **MAE 3.58 / ρ 0.766**. A size-matched control indicates this is **not explained by
sample count alone**.

**Restoration is asymmetric and architecture-dependent.** Shifts that alter *geometry* teach a
specific, powerful cue that only other geometric shifts can use (ρ 0.73–0.86); shifts that add
*noise* teach a weaker but broader lesson (ρ 0.39–0.54). We identify the carrying feature and
show it is inert on noise-driven shifts. Across architectures, the in-distribution result
replicates in a world model with **no recurrence and no autoregressive latent feedback**, but
the restoration effect does not — it scales with how badly the underlying model degrades.

We report all nine pre-registered hypotheses, including five that were not supported, a failed
adaptive-planning experiment, and two configurations rejected as invalid instruments by a
pre-registered gate.

---

## 1. Introduction

World models let an agent plan by imagining. Every such agent faces a question it rarely asks
explicitly: *for how long is this imagination worth acting on?* Too short a horizon wastes the
model; too long, and the agent plans against a hallucination.

The standard answer thresholds an uncertainty signal: roll forward, stop when disagreement or
entropy crosses a bound. Prior work establishes the premise this rests on — epistemic
uncertainty correlates with realised rollout error. We call this **Claim A**.

We study a stronger claim. **Claim B**: an agent can learn a *monitor* that predicts the horizon
itself, per state, as a quantity — not merely rank states by riskiness. The distinction matters
because a threshold rule is only as good as the monotone map from uncertainty to trust; if that
map is all there is, nothing beyond Claim A has been shown.

Our central methodological device is therefore a control, not a baseline. For each uncertainty
scalar we fit an isotonic regression from that scalar to `H*` **on the test set itself** — the
best possible performance of *any* monotone function of that signal, with no generalisation
gap. Beating it is the minimum evidence that Claim B says something Claim A does not.

Having established the monitor works in distribution, we ask when it stops working, whether
simple fixes recover it, and what kind of experience does. The negative results are the
substance: the contribution is not another uncertainty metric but a characterisation of *when
imagination monitoring works, when it fails, and what restores it*.

### Contributions

1. **A per-state horizon-prediction task** with a discrete, exactly-checkable target and an
   optimistic monotone ceiling as the discriminating control (§3, §5.1).
2. **Evidence for Claim B in distribution**, with the finding that the monitor **requires
   inspecting the imagined rollout** — the strictly a-priori version fails (§5.1).
3. **An anti-conservative out-of-distribution failure**, shown to be a loss of
   rank-informativeness rather than of scale or calibration (§5.2–5.3).
4. **Restoration of transfer by exposure to a shifted distribution**, with a size-matched
   control ruling out sample count as the explanation (§5.4).
5. **Characterisation of that restoration as asymmetric**, with an identified carrying feature
   that is useless in distribution, dominant on geometric shifts, and inert on noise-driven
   ones (§5.5).
6. **Architecture evidence**: the in-distribution result replicates in a non-recurrent
   Transformer; the monitor is only partially portable between architectures; and the
   restoration effect is architecture-dependent (§5.6).
7. **Full reporting of negative results**, including adaptive planning and two invalid
   instruments (§5.7, §6).

---

## 2. Related Work

| Work | What it establishes | How this differs |
|---|---|---|
| Certified World Models (2026) | A-priori horizon from Jacobian/Lyapunov spectrum, `T(ε) ~ log(1/ε)/λ` | Orbit-wide and worst-case; the authors state it does not predict *which* state breaches tolerance. We monitor per-state and evaluate as a prediction task. |
| Conformal Trust Horizons (2026) | Trust horizon from one-step residual + expansion, conformally calibrated | Analytic, requires a known symmetry group; a distributional audit, not a learned per-state monitor. |
| Round-Trip Consistency (2026) | Models predict their own rollout error (ρ 0.91–0.98) | Requires a forward+backward rollout and an invertible model; PDE/video, not an acting agent. Our §5.1 result narrows this gap: our monitor also needs *a* rollout, though a cheaper one. |
| Biased Dreams (2026) | **Negative**: latent ensemble disagreement does not track compounding error; attractor collapse | Predicted our failure. We confirm the mechanism for predictive entropy but *not* for ensemble spread, and locate the real problem in rank-informativeness (§5.3). |
| MBPO / MACURA / adaptive rollout | Truncate the rollout when uncertainty crosses a threshold | Online truncation observes the rollout as it unfolds. Included as a baseline; our monitor beats it by 1.96 MAE. |
| Uncertainty-aware world models for MBRL | Epistemic uncertainty tracks realised error | This is Claim A — our weak control, not our result. |

The gap we occupy: a **learned, per-state monitor**, evaluated against ground truth with an
optimistic monotone control, and characterised under distribution shift and across
architectures.

---

## 3. Problem Formulation

For a world model `f_θ` and state `s_t`, let `ŝ_{t+H}` be the imagined state after `H` steps
under a known action sequence and `s_{t+H}` the realised state, with `E_H = d(ŝ_{t+H},
s_{t+H})`:

```
H*(s_t) = max { H ≤ H_max : E_h < ε for all h ∈ 1..H }
```

We use the **first-failure** definition: once imagination diverges it is operationally unusable
even if it re-synchronises coincidentally. `H_max = 30`.

**Environment and metric.** MiniGrid with 5×5 egocentric symbolic observations. The world model
predicts size-agnostic state *factors* — direction, whether the transition moved the agent,
carrying, door state — and position is reconstructed by integrating `moved` along `dir`. `E_H`
is the Manhattan error of that integrated displacement; `ε = 1`, so `E_H < ε` means the model
places the agent in **exactly** the right cell. The integration identity is verified exactly on
every configuration before any modelling.

This makes the object of study concrete. The load-bearing quantity is `moved` — whether the
cell ahead is free — which is exactly what partial observability hides. **Imagination stays
reliable roughly as long as the planned path stays inside what the agent has already seen.**

**The monitor.** `g_φ` maps time-`t` information to a distribution over `H* ∈ {0..30}`, giving a
point estimate and `P(H* ≥ h)` for calibration. It may see current and past observations, the
belief state, uncertainty signals, and the planned action sequence. It may **not** see any
observation, state, or error at time > `t`. This boundary is verified executably: corrupting
every future field leaves all features bit-identical while `H*` moves from 6.45 to 0.01.

**Configurations.** A: `DoorKey-16x16` (training). B: `FourRooms`. C: A + 15% action slip.
D: `MultiRoom-N6`. E: `SimpleCrossingS11N5`. F/F2: observation noise. F3: 20% action dropout.
Models and monitors are fitted on A unless stated.

---

## 4. Protocol

All of the following was frozen before the corresponding evaluation, across five separate
pre-registration documents; departures are logged with timing and reasons.

- **Splits.** Layout seeds disjoint across world-model training, monitor training, validation
  and test. The monitor trains on episodes the world model never saw, so `H*` is not inflated
  by memorisation.
- **Statistics.** Episode-clustered bootstrap, 2000 resamples, 3 training seeds, 5-member
  ensembles, paired differences throughout.
- **Model selection** on validation only; test data never informs any choice.
- **Degeneracy gate.** `H*` with IQR < 2 or > 80% at ceiling ⇒ the configuration cannot test
  anything and is declared inconclusive. Checked on validation before test.
- **Validity gate** (added for H9 after H8 exposed the need). A new configuration counts as an
  instrument only if `H*` is ≥ 40% below A's *and* IID-only transfer to it is < 0.35.

---

## 5. Results

### 5.1 Monitoring works in distribution — but not a priori

`H*` on A-test: mean 15.52, sd 10.29, IQR 20.0, 21.2% at ceiling. Non-degenerate.

| Method | MAE ↓ | Spearman ↑ | R² ↑ |
|---|---|---|---|
| **Monitor `g_φ`** | **6.42 ± 0.07** | **0.573 ± 0.005** | **0.331** |
| **Claim-A ceiling** *(isotonic, fitted on test)* | 7.46 ± 0.13 | 0.456 | 0.218 |
| Monitor, **strict a-priori** (no imagined rollout) | 8.07 ± 0.09 | 0.372 | 0.119 |
| Hand-crafted + actions (no model internals) | 8.27 ± 0.10 | 0.320 | 0.107 |
| Online truncation (MBPO/MACURA-style) | 8.38 ± 0.08 | 0.298 | 0.096 |
| Privileged: true local grid + ordered actions | 8.63 ± 0.07 | 0.226 | 0.030 |
| Fixed horizon (median) | 8.98 ± 0.10 | 0.000 | −0.010 |

The monitor beats the constant predictor by 2.51–2.63 MAE and the optimistic ceiling by
**0.94–1.10 MAE**, with bootstrap CI lower bounds of 0.978 / 1.010 / 0.854 across seeds.
Calibration is good in distribution: ECE 0.023–0.041 for `P(H* ≥ 3/5/10/20)`.

**The strict a-priori variant fails** (−0.46 / −0.63 / −0.74 against the ceiling). Restricted to
belief state, one-step uncertainty, planned actions and observation statistics, it does not beat
a monotone function of a single scalar. The signal lives in the *imagined trajectory's internal
disagreement*, so the honest reading of "before observing the future" is: before observing it,
but **after imagining it**.

A **privileged** predictor given the true local 11×11 grid and the ordered action sequence does
*worse* (ρ 0.226) than the monitor. Knowing the environment is not sufficient; the monitor must
know the *model's* relationship to it.

### 5.2 Distribution shift breaks the monitor, anti-conservatively

| Split | actual `H*` | predicted | ρ |
|---|---|---|---|
| A (in-dist) | 15.52 | 15.55 | **0.573** |
| B (unseen layouts) | 7.75 | **21.56** | 0.199 |
| C (action slip) | 4.11 | **13.59** | 0.238 |
| D (corridors) | 5.11 | **21.15** | 0.128 |

Nearly unbiased in distribution; under shift the true horizon collapses while the prediction
*rises*. The monitor is most confident where the model is worst — the opposite of the property
a safety-relevant component needs, and a more damaging failure than ordinary calibration decay.

### 5.3 Simple repairs fail: this is rank-informativeness, not scale

Two repairs, neither fitted on OOD data: rank-normalising the uncertainty features, and a
novelty gate calibrated to be inert in distribution. **Both fail on all seeds**; predictions
remain ≈20 against true horizons of 4–8.

The diagnostic explains why, and refutes the mechanism we pre-registered:

| feature | ρ_A | ρ_B | ρ_C | ρ_D |
|---|---|---|---|---|
| `imagined_disp_spread` | **−0.455** | −0.096 | −0.179 | −0.096 |
| `ent_moved_mean` | −0.40 | **−0.16** | −0.11 | **+0.04** |

Ensemble disagreement moves in the *right* direction OOD (mean spread 0.53 → 1.0–1.5), refuting
a pure attractor-collapse reading for that signal; predictive entropy moves in the *wrong*
direction, confirming it for that one. The decisive problem is neither: **every feature loses
per-state discriminative power**. The signal still reports "harder here" on average; it no
longer reports *which state* will fail. Rank-normalisation therefore preserves an ordering that
has itself become uninformative.

### 5.4 Exposure to a shifted distribution can restore transfer

Refitting `g_φ` on combinations of configurations (no world-model retraining; each OOD
configuration split 60/20/20 with the test portion evaluation-only in every regime):

| regime | test A | test B | test C | test D |
|---|---|---|---|---|
| A | 0.580 | **0.218** | **0.213** | **0.135** |
| A+B | 0.584 | 0.755 | **0.306** | **0.771** |
| A+C | 0.576 | **0.461** | 0.479 | **0.510** |
| A+B+C | 0.577 | 0.754 | 0.477 | **0.766** |

On **D, never seen in training**, `A+B+C` moves MAE 16.77 → **3.58** and ρ 0.135 → **0.766**,
with the prediction becoming properly conservative (21.2 → 5.2 against a true 4.89).

A **size-matched control** — every regime subsampled to the A-only anchor count — reproduces the
effect (MAE 3.62, ρ 0.762). This **indicates the effect is not explained by sample count
alone**; it does not establish diversity as the only causal factor, and we do not claim it does.
In-distribution performance is preserved in every regime.

Adding a second shift buys little: `A+B → D` (0.771) matches `A+B+C → D` (0.766). Within this
environment family, transfer can be **substantially restored by exposure to an appropriate
shifted distribution**, and the marginal value of further shifts is small. §5.6 shows the effect
is not universal across architectures.

### 5.5 Restoration is asymmetric, and we identify the carrier

Tested predictively on two new configurations (E, geometric; F3, noise-driven), pooled with the
originals to give four donors and four valid recipients:

| | → geometric (B, D, E) | → noise-driven (C, F3) |
|---|---|---|
| **geometric donor** (B, E) | **+0.73 … +0.86** | +0.31 |
| **noise donor** (C, F3) | +0.39 … +0.54 | +0.44 … +0.47 |

An advance prediction was confirmed on the geometric side: `A+B → E` = **+0.855** on a
never-seen maze family (3/3 seeds, baseline +0.116), and E functions as a donor (`A+E → D` =
+0.800). The noise side did **not** meet its pre-registered bar (`A+C → F3` = +0.444 vs 0.50).
We therefore **withdraw** the symmetric two-family (epistemic/aleatoric) account we had
proposed, and report the asymmetry that the data support.

**Mechanism.** The carrier is `free_ahead` — how far the agent can currently see clear space:

| | A | B | C | D | E | F2 | F3 |
|---|---|---|---|---|---|---|---|
| ρ(`free_ahead`, `H*`) | +0.004 | +0.586 | +0.024 | +0.566 | +0.533 | −0.015 | −0.031 |

It is **useless in distribution** (+0.004) and **dominant on geometric shifts** (+0.53–0.59), so
a monitor trained only on A has no reason to use it and shift exposure teaches it to. It is
**inert on every noise-driven configuration**. No equivalent single carrier exists on the noise
side, which is why those donors transfer broadly but never strongly.

No feature is both stable and useful: the lowest rank-instability belongs to a feature with
ρ ≈ 0 everywhere. There is no distribution-invariant uncertainty scalar here — which is why a
*learned* monitor with shift exposure is needed rather than a better hand-picked signal.

### 5.6 Partial architecture transfer, and an architecture-dependent restoration effect

| architecture | A MAE | A ρ | beats its own ceiling |
|---|---|---|---|
| GRU-RSSM (deter 512) | 6.408 | +0.584 | **yes** (+1.355) |
| GRU-RSSM (deter 256) | 6.351 | +0.551 | **yes** (+1.050) |
| **Transformer** (no recurrence, no latent feedback) | 6.432 | +0.513 | **yes** (+0.993) |

The in-distribution result is **not a recurrent-architecture artefact**: a model with no
recurrence and no autoregressive latent feedback reproduces it at comparable strength, each
measured against its own ceiling on its own `H*`.

Applying a monitor fitted on one architecture to another gives a clear hierarchy:
within-architecture 0.51–0.58, within-family 0.36–0.39, cross-family 0.21–0.36. The monitor is
**partially portable** — clearly above zero, but it loses a third to two-thirds of its signal
and cannot be reused as-is.

Critically, the §5.4 restoration effect **does not replicate in the Transformer**
(0.257 → 0.290). The reason is instructive: the Transformer is a much better world model OOD
(`H*` 8.55 on D vs ≈5.1; only 3.8% of anchors collapse to zero vs ≈29%). For the GRUs, shift
exposure largely teaches "out of distribution, expect near-zero" — a large, easily-learned
correction. **The magnitude of the restoration effect scales with how badly the underlying
model degrades**, so a stronger world model needs it less. This is the second half of why the
headline claim is bounded: the monitor is distribution-dependent, and what restores it is
architecture-dependent.

### 5.7 Adaptive planning does not help (weak test)

| arm | success | imagined states/decision |
|---|---|---|
| fixed H=5 (best) | **0.933 ± 0.032** | 480 |
| fixed H=30 | 0.367 ± 0.062 | 2880 |
| adaptive `Ĥ*` | 0.517 ± 0.065 | 2637 |

Adaptive is worse (−0.417 success) and 5.49× more expensive. But success falls *monotonically*
with horizon, so the compute–performance frontier has **no interior optimum** and there is
nothing for an adaptive horizon to find; our attempt to make the task non-myopic failed (mean
detour 0.1 cells). We report this as a **weak test**, not a refutation. The adaptive arm's
specific failure — selecting H ≈ 26 where 3–5 is optimal — is the same anti-conservative bias as
§5.2, reappearing under the state distribution shift that MPC itself induces.

---

## 6. Limitations

- **One environment family.** MiniGrid with symbolic observations, ~2.8M-parameter models. The
  mechanism (visible horizon vs planned path) may not survive pixels or continuous control, and
  `free_ahead` is domain-specific. The general form of the claim — *a cue exists that is useless
  in distribution and dominant under shift* — is what would need to transfer.
- **Right-censoring.** 21% of A-test targets sit at `H_max = 30`, inflating MAE and leaving the
  upper tail unidentified.
- **Modest absolute accuracy.** R² = 0.33; most variance is unexplained. The margin over the
  optimistic ceiling is ~1 MAE against a target with sd 10.3 — statistically solid, modest in
  absolute terms.
- **Asymmetry rests on four donors**, two per side, within one environment family.
- **Architecture study is single-seed** (K=3) — an existence check, not a precision estimate.
- **Two configurations were invalid instruments.** Observation noise never degraded the model
  enough to test anything, even at 70% corruption. The noise side of §5.5 rests on one strong
  instrument (action dropout) and one pre-existing one (slip).
- **Two scoring-code/protocol mismatches** were found and are reported in the results rather
  than silently resolved; neither changed a verdict.

---

## 7. Conclusion

```
Monitoring works  →  OOD breaks it  →  simple repairs fail
                  →  shift exposure can restore it
                  →  restoration is asymmetric and architecture-dependent
```

World models can learn a useful monitor of the reliability horizon of their imagined futures.
That monitor beats the strongest control we could construct — the best monotone function of any
single uncertainty scalar, fitted optimistically on the test set. But it must imagine before it
can judge; it fails anti-conservatively under distribution shift; that failure resists
rescaling and novelty gating because the signals lose per-state ordering rather than scale; and
while exposure to an appropriate shifted distribution can substantially restore transfer, what
gets restored depends on *which kind* of shift was seen and on *how badly* the underlying model
degrades.

The practical implication is uncomfortable for uncertainty-gated imagination: a monitor trained
where the model works will be most confident where the model fails, and cannot be fixed by
rescaling. It can be improved by training on failures — but only failures of the right kind, and
less so when the world model is already strong.

**Experimental ledger.** 9 pre-registered hypotheses, 5 frozen protocols, 24 world models across
3 architectures. Supported: H1, H2, H6 (H7 partial). Not supported: 5. Every negative retained.

---

## Appendix

- `PREREGISTRATION.md`, `PREREGISTRATION_H5/H6/H8/H9.md` — frozen protocols
- `DEVIATIONS.md` — every departure, with timing relative to test evaluation
- `RESULTS.md` — full tables, per-seed statistics, figures
- `tests/test_leakage.py` — executable verification of the information boundary
- Reproduction: `scripts/run_experiment.py`, `analyze.py`, `run_transfer.py`,
  `run_cross_arch.py`, `run_taxonomy.py`, `run_aleatoric.py`
