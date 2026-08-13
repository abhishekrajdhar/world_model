# Pre-Registration: H9 — Does the aleatoric family transfer? (a valid instrument this time)

**Status: FROZEN** before configs F2/F3 were collected. H8 is final and is **not** modified by
anything here; this is a separate experiment.

## Why H9 exists

H8 confirmed the **epistemic** half of the taxonomy predictively (`A+B → E` = +0.855 on an
unseen maze family, 3/3 seeds; `free_ahead` +0.533 on E). The **aleatoric** half was never
tested, because config F (15% observation-cell corruption) failed as an instrument:

| | `H*` | IID-only transfer (A → cfg) |
|---|---|---|
| A (in-dist) | 15.52 | 0.580 |
| **F (failed instrument)** | **15.23** | **+0.472** |
| B | 7.75 | +0.218 |
| C | 4.11 | +0.213 |
| D | 5.11 | +0.135 |

F barely degraded the world model, so predictions about transfer *to* it were vacuous. C
remains the only member of the aleatoric family.

## Validity gate — the fix for what went wrong, specified in advance

A new config counts as a **valid instrument** only if **both** hold, measured before any
prediction is scored and independent of transfer outcomes:

1. `H*` is at least **40% below** A's (i.e. `H* <= 9.3`), and
2. IID-only transfer `A → config` is **< 0.35**.

A config failing either gate is reported as an **invalid instrument** and its predictions are
declared uninformative — exactly the verdict F should have received by rule rather than by
argument. Had this gate existed, F would have failed condition 1 (15.23) and condition 2
(0.472).

## New configs — two mechanisms, both aleatoric, both distinct from C

C perturbs the *executed action* (substitution). The new ones perturb different things:

- **F2 — perception noise.** Corrupt a fraction of object-channel cells in the egocentric
  view. Same family as F but at a severity chosen to bite.
- **F3 — action dropout.** With probability `p` the requested action is replaced by MiniGrid's
  `done` action, a verified true no-op (`minigrid_env.py`: `elif action == self.actions.done:
  pass`), so the agent's intent silently fails to execute. Distinct from slip, which
  substitutes a *different* action.

### Severity calibration rule (fixed in advance, outcome-independent)

Ladders: F2 `p in {0.30, 0.50, 0.70}`, F3 `p in {0.20, 0.35, 0.50}`.

For each candidate, collect **200 calibration episodes** on layout seeds `[200000, 200200)` —
disjoint from the `[100000, 100600)` seeds used for every scored test set — and measure `H*`
using the **seed-0** models only. Select the rate whose mean `H*` is **closest to config B's
7.75**, a target fixed by an existing measurement rather than by any H9 outcome. Ties break
toward the lower rate. All ladder values are reported.

## Frozen predictions

Scored with the H6 protocol: `no_latent` features, 60/20/20 episode-id split, evaluation-only
test portion, standardiser fitted on training only, 3 seeds, episode-clustered bootstrap.

| # | cell | threshold | tests |
|---|---|---|---|
| **Q1** | `A+C → F2` | ρ > 0.50 | **decisive**: aleatoric donor → different aleatoric mechanism |
| **Q2** | `A+B → F2` | ρ < 0.35 | epistemic donor does *not* rescue an aleatoric recipient |
| Q3 | `A+F2 → C` | ρ > 0.50 | reverse direction |
| Q4 | `A+C → F3` | ρ > 0.50 | substitution → dropout |
| Q5 | `A+B → F3` | ρ < 0.35 | epistemic donor → aleatoric recipient (second mechanism) |
| Q6 | `free_ahead` on F2, F3 | \|ρ\| < 0.15 | the epistemic carrier should stay uninformative |

## Verdict rule

Evaluated only over configs that pass the validity gate.

- **SUPPORTED** — Q1 and Q2 both hold on >= 2 of 3 seeds. The aleatoric family is real *and*
  distinct from the epistemic one, and the taxonomy stands as a two-family claim.
- **PARTIALLY SUPPORTED** — Q1 holds but Q2 fails: transfer within the aleatoric family is
  real, but epistemic donors help too, so "family" is not the organising principle —
  *any* sufficiently severe shift may suffice. This would replace the taxonomy with a
  **severity** account, which we would state as such.
- **NOT SUPPORTED** — Q1 fails on a valid instrument: aleatoric shifts do not transfer to each
  other, and only the epistemic grouping survives. C would then be reported as sui generis.
- **INCONCLUSIVE** — no new config passes the validity gate.

## Compute

No world-model retraining: existing main-run checkpoints, data collection plus one
feature-extraction pass per config. Calibration uses a reduced anchor cap and seed 0 only.
This is the last planned experiment; after it the experimental phase is frozen.
