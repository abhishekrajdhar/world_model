# Pre-Registration: H8 — Is metacognitive transfer organised by failure-mode family?

**Status: FROZEN** before configs E and F were collected. No E/F data existed when this was
written.

## The claim under test

H6 found that horizon prediction transfers between B (unseen layouts) and D (long corridors)
but not to or from C (action slip), and explained it mechanically: `free_ahead` — how far the
agent can currently see — carries the OOD signal for B (rho +0.586) and D (+0.566) but is
useless for C (+0.024).

We labelled that split **epistemic** (B, D: uncertainty about unseen geometry) versus
**aleatoric** (C: irreducible noise). That taxonomy is currently inferred from three configs,
with C the sole member of its class. It is a post-hoc description fitted to a 2-vs-1 split.

H8 converts it into a **predictive** claim that can fail.

## New configs

- **E — epistemic.** `MiniGrid-SimpleCrossingS11N5-v0`: a new maze topology, no added noise.
  A geometric shift the world model has never seen.
- **F — aleatoric.** Config A with **observation noise**: 15% of cells in the egocentric view
  are corrupted to a random object id. Dynamics are untouched. This is a *different aleatoric
  mechanism* from C — unreliable perception rather than unreliable action — which is what
  makes it a real test rather than a re-run of C.

Both are evaluated with the **existing main-run world models** (seeds 0–2, K=5). No
retraining: observations are egocentric, so the model applies to any layout by construction.
E and F are collected with the same policy, `H*` definition, `epsilon`, and `H_max`.

## Registered predictions (made before the data exists)

Let rho(train -> test) be Spearman on the held-out config's test portion, using the frozen
`no_latent` features and the H6 protocol (60/20/20 episode-id split, evaluation-only test
portion, standardiser fitted on training only, model selection on training-config validation).

| # | Prediction | Threshold |
|---|---|---|
| **P1** | `A+B -> E` **transfers** (epistemic donor, epistemic recipient) | rho > 0.5 |
| **P2** | `A+B -> F` **does not transfer** (epistemic donor, aleatoric recipient) | rho < 0.35 |
| **P3** | `A+C -> F` **transfers** (aleatoric donor, aleatoric recipient, different mechanism) | rho > 0.5 |
| **P4** | `A+E -> D` **transfers** (E behaves as an epistemic donor) | rho > 0.5 |
| **P5** | `A+F -> C` **transfers** (F behaves as an aleatoric donor) | rho > 0.5 |
| **P6** | `free_ahead` is informative on E and near-zero on F | \|rho_E\| > 0.3, \|rho_F\| < 0.15 |

**P3 is the decisive one.** C and F share no mechanism — one perturbs actions, the other
perturbs perception. If metacognitive transfer really is organised by *failure-mode family*,
C must nonetheless teach the predictor something usable on F. If P3 fails while P1 holds, the
organising principle is **not** epistemic-vs-aleatoric but something narrower — most plausibly
"which feature happens to carry the signal" — and the taxonomy claim is withdrawn.

## Verdict rule

- **SUPPORTED** — P1, P2 and P3 all hold, on at least 2 of 3 seeds each.
- **PARTIALLY SUPPORTED** — P1 and P2 hold but P3 fails: transfer is real but organised by
  something narrower than failure-mode family. The taxonomy language is withdrawn from the
  paper and replaced by the feature-level account.
- **NOT SUPPORTED** — P1 fails: the epistemic grouping does not predict transfer to a new
  epistemic config, and the H6 pattern is specific to B/D rather than general.

P4–P6 are secondary and diagnostic; they do not change the verdict.

## Controls carried over from H6

Same architecture, features, target, splits protocol, seeds and statistics (episode-clustered
bootstrap, 2000 resamples, 3 seeds). The **size-matched control** is run again: any regime
with more training data than the A-only baseline is also evaluated subsampled to the A-only
anchor count. In-distribution A-test MAE must not degrade by more than 0.5 in any regime.

## What would make this uninterpretable

If `H*` on E or F is degenerate (IQR < 2, or > 80% of anchors at the `H_max` ceiling, or
> 80% at zero), that config cannot test anything and is reported as INCONCLUSIVE rather than
as evidence either way. This check is run and reported before the predictions are scored.
