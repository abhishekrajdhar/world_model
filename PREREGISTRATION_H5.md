# Pre-Registration: H5 — Repairing the OOD failure of the horizon predictor

**Status: FROZEN** before any of this experiment's numbers were computed. Written after the
main run, so the *motivating* result (§8 of `RESULTS.md`) is known; the outcome of every arm
below is not. No diagnostic on the OOD feature statistics was inspected before freezing.

## Motivation

The main run established H2 (a learned per-state horizon predictor beats the best monotone
function of any single uncertainty scalar) but refuted H3: out of distribution `g_phi`
predicts **longer** horizons while the true horizon collapses.

| Split | actual `H*` | predicted `H*` | Spearman |
|---|---|---|---|
| A (in-dist) | 15.52 | 15.55 | 0.573 |
| B (unseen layouts) | 7.75 | 20.01 | 0.13 |
| C (stochastic) | 4.11 | 12.82 | 0.22 |
| D (long corridors) | 5.11 | 19.97 | 0.12 |

## Competing explanations (the thing this experiment is designed to separate)

- **E1 "mis-scaled".** The uncertainty features do move in the right direction OOD, but
  `g_phi` maps them through a response fitted to config A's magnitude range, so the mapping
  saturates or extrapolates wrongly. *Predicts: rank-normalising the features repairs it.*
- **E2 "actively misleading".** The features move in the **wrong** direction OOD. In cluttered
  layouts the world model predicts "blocked, no movement", all ensemble members agree, so
  disagreement and imagined-displacement spread are *low* precisely where the model is worst
  (the attractor collapse of [Biased Dreams](https://arxiv.org/abs/2604.25416)).
  *Predicts: rank-normalisation fails, and only an explicit novelty signal can help.*

E1 and E2 make opposite predictions about arm R1, so the experiment is diagnostic either way.

## Arms

All arms train `g_phi` on **A-gtrain only** and select on **A-val only**. Nothing is fitted on
B/C/D. All use the `no_latent` feature variant (best in the main run) for comparability.

- **R0 — baseline.** Reproduces the main-run predictor. Control.
- **R1 — rank-normalised.** Every `apriori` + `imag` (uncertainty) feature is replaced by its
  quantile rank computed **within the split being evaluated**. Tests E1.
- **R2 — novelty-gated.** R0's prediction is shrunk by
  `Ĥ_adj = Ĥ * exp(-lambda * max(0, nu - tau))`, where `nu` is `latent_nn_dist` (distance from
  the anchor latent to the nearest A-gtrain latent). `tau` = 95th percentile of `nu` on
  **A-val**; `lambda = -ln(0.9) / (q99 - q95)` on A-val, i.e. fixed so an in-distribution
  state at the 99th novelty percentile is shrunk by exactly 10%. Both constants come from
  A-val alone and are set by this rule, not tuned.
- **R3 — combined.** R1 features plus the R2 gate.

## Frozen success criteria

**H5 SUPPORTED** if, for at least one arm among R1/R2/R3, all three hold:

1. mean `Ĥ*` is **lower than on A-test** for **all three** OOD configs;
2. Spearman(`Ĥ*`, `H*`) **> 0.3** on **at least two** of the three OOD configs;
3. A-test MAE degrades by **≤ 0.5** relative to R0 (the in-distribution result must not be
   traded away).

**PARTIALLY SUPPORTED** if (1) and (3) hold but (2) fails — the predictor becomes
appropriately conservative but loses per-state ordering.

**NOT SUPPORTED** if no arm satisfies (1) and (3).

Reported with the same statistics as the main run: episode-clustered bootstrap 95% CIs over
2000 resamples, averaged across the 3 training seeds, per split.

## Diagnostic (reported, not a hypothesis test)

Per-split means of each uncertainty scalar, plus its Spearman correlation with `H*` computed
**within** each split. This directly adjudicates E1 vs E2: if a feature's within-split
correlation keeps its sign OOD but its magnitude range shifts, that is E1; if the sign flips
or the mean moves opposite to difficulty, that is E2.

## Honest limitation, stated in advance

R1's within-split ranking is **transductive**: it uses the distribution of the evaluation
split. A deployed agent would approximate it with a rolling buffer of its own recent states.
If R1 succeeds we must therefore report it as an upper bound on what a causal implementation
would achieve, and the causal version becomes the next experiment rather than a claim made
here.
