# Pre-Registration: H6 — Is simulation metacognition transferable across distribution shifts?

**Status: FROZEN** before any H6 number was computed. The motivating results (H3, H5) are
known; every H6 outcome below is not.

## Question

H5 established that the OOD failure is **not** an input-scaling problem. The uncertainty
features still separate distributions *globally* (mean `dispspread` 0.53 in-distribution vs
1.0–1.5 OOD) but lose the ability to rank *which individual states* will fail (within-split
Spearman with `H*` collapses from −0.46 to −0.09…−0.20).

H6 asks whether that ordering can be restored by **exposure to diverse failure modes during
training of `g_phi`** — with no world-model retraining.

## What is explicitly NOT claimed

If transfer fails we will **not** conclude that reliable-horizon prediction is
distribution-specific by nature. Failure is consistent with insufficient training diversity,
insufficient predictor capacity, covariate shift, feature representation, optimisation, or
sample size. The registered claim is only:

> We test whether exposure to diverse shifts restores cross-distribution horizon prediction.

The conclusion will be determined by the result, and the confounds above will be listed
alongside any negative finding.

## Design

Five training regimes, three diversity levels:

| Regime | Train configs | Level |
|---|---|---|
| `A` | A | **R0** — IID only |
| `A+B` | A, B | R1 — one shift |
| `A+C` | A, C | R1 — one shift |
| `A+D` | A, D | R1 — one shift |
| `A+B+C` | A, B, C | **R2** — multiple shifts |

Every regime is evaluated on the held-out test portion of **all four** configs, giving the
transfer matrix. `A+B+C → D` is the critical cell: a completely unseen shift under maximum
training diversity.

## Splits (new, and the main contamination risk)

The main run produced only *test* features for B/C/D. To train on a shift config without
contaminating its evaluation, each OOD config's episodes are partitioned **by episode id**,
deterministically:

- `ep_id % 5 in {0,1,2}` → train (60%)
- `ep_id % 5 == 3` → val (20%)
- `ep_id % 5 == 4` → **test (20%) — evaluation only, never trained on in any regime**

Config A keeps its existing `A_gtrain` / `A_val` / `A_test` splits.

**Every regime is evaluated on the same fixed test portions**, whether or not that config
appeared in its training set. A config in the training set is reported as a *seen* shift; one
absent from it is *unseen*. Because OOD test portions are 20% of 600 episodes (~120 episodes,
~2.2k anchors), CIs will be wider than the main run's and we report them.

## Frozen protocol

- **Architecture:** `HorizonPredictor`, hidden 256, depth 2 — unchanged from the main run.
- **Features:** the `no_latent` variant, unchanged. No new features.
- **Target:** `H*` first-failure definition, `epsilon = 1`, `H_max = 30`, unchanged.
- **Normalisation:** the standardiser is fitted on the **training split only**. No test-set
  normalisation, and no rank-normalisation (H5 showed it does not help and it is transductive).
- **Model selection:** on the union of the **validation** portions of the training configs
  only. Test data is never consulted for any choice.
- **Seeds:** the same three world-model seeds (0, 1, 2); the predictor uses the same seed.
- **Class balance:** training sets differ in size across regimes; we report the anchor count
  per regime, and additionally report a **size-matched** control in which multi-config
  regimes are subsampled to the size of the A-only regime, so any gain cannot be attributed
  to simply having more training data.

## Metrics

Per (regime, evaluation config): MAE, RMSE, Spearman, R², mean predicted vs actual `H*`, and
ECE for `P(H* >= h)` at h in {3, 5, 10, 20}. Episode-clustered bootstrap 95% CIs, 2000
resamples, averaged over 3 seeds.

**Transfer metric.** For held-out config `j` and regime `i`:

```
T_MAE(i -> j) = MAE(g_A, j) - MAE(g_i, j)      # positive = improvement over the IID-only baseline
T_rho(i -> j) = rho(g_i, j) - rho(g_A, j)
```

reported as a matrix with paired bootstrap CIs.

## Frozen success criteria

For a regime `i` and a **completely held-out** config `j` (j not in i), transfer is
**RESTORED** if all three hold:

1. mean predicted `Ĥ*` on `j` is **lower** than the same regime's mean prediction on A-test
   (conservativeness restored — this is the criterion the main run failed);
2. Spearman on `j` **> 0.3**;
3. `T_MAE(i -> j) > 0` with paired bootstrap 95% CI excluding 0.

Overall verdict:

- **SUPPORTED** — transfer is RESTORED for the critical cell `A+B+C → D`, on all three seeds.
- **PARTIALLY SUPPORTED** — restored for at least one unseen (regime, config) pair but not
  the critical cell. This is the *metacognitive transferability* outcome: transfer between
  related failure modes but not across qualitatively different ones. If so we report which
  shift pairs transfer.
- **NOT SUPPORTED** — no unseen cell satisfies (1)–(3), with the confounds above listed.

Additionally, in-distribution performance must be preserved: A-test MAE for any regime must
not degrade by more than 0.5 vs the A-only regime. A regime that buys OOD transfer by
destroying the ID result is reported as such, not as a success.

## Rank-stability analysis (reported, not a hypothesis test)

For every scalar uncertainty feature `x` and every distribution `D` in {A, B, C, D}, compute
`rho(x, H* | D)` on that config's test portion, then

```
RS(x) = Var_D[ rho(x, H* | D) ]          # rank instability
```

reported next to `|rho_A|`. A feature with strong in-distribution predictive power but high
`RS` is exactly the pathology H5 identified, and this quantifies it per feature. We also
report whether any feature has low `RS` — such a feature would be the natural basis for a
transferable horizon predictor and would define the next experiment.

## Compute

No world-model retraining. Runs on the precomputed features from the main run
(`rih-features-main` dataset), on Kaggle.
