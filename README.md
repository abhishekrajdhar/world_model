# Reliable Imagination Horizon (RIH)

**Can an agent know when its imagined world is wrong?**

We test whether a world-model agent can estimate, from information available at time `t`
only, how many steps of its own imagined future will remain trustworthy — *before* observing
that future.

```
H*(s_t) = max { H <= H_max : E_h = 0 for all h in 1..H },    E_H = d(s_hat_{t+H}, s_{t+H})
```

The question is whether `U_t -> H*(s_t)`, and specifically whether a learned predictor beats
the best *monotone function of any single uncertainty scalar* — because that weaker claim
("uncertainty correlates with error") is already established in the literature.

## Start here

| File | What it is |
|---|---|
| [PREREGISTRATION.md](PREREGISTRATION.md) | Hypotheses, thresholds, splits, decision rules — **frozen before any test evaluation** |
| [docs/phase0_literature.md](docs/phase0_literature.md) | Literature audit and the exact novelty boundary |
| [RESULTS.md](RESULTS.md) | Final report (generated after the main run) |
| [DEVIATIONS.md](DEVIATIONS.md) | Every departure from the pre-registration, with reasons |

## The mechanism being tested

Observations are egocentric 5x5 symbolic views, so the agent sees roughly 4 cells ahead.
The world model predicts size-agnostic state *factors* (`dir`, `moved`, `carrying`,
`door_open`) and position is reconstructed by integrating `moved` along `dir`. The
load-bearing quantity is **`moved`** — whether the cell ahead is free — because that is
exactly what partial observability hides.

This makes the reliable horizon mean something concrete: **imagination stays exact roughly as
long as the planned path stays inside the region the agent has already seen.**

## Environments

| Config | Env | Role |
|---|---|---|
| A | `DoorKey-16x16` | training distribution |
| B | `FourRooms` | unseen layout topology |
| C | `DoorKey-16x16`, slip=0.15 | stochastic dynamics (aleatoric) |
| D | `MultiRoom-N6` | long corridor chains |

Everything (world model and `g_phi`) is fitted on **A only**; B/C/D are OOD evaluation.
Layout seeds are disjoint across `wmtrain / gtrain / val / test`, and the horizon predictor
is trained on episodes the world model never saw.

## Reproducing

```bash
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

```bash
./.venv/bin/python scripts/run_experiment.py --config pilot
```

```bash
./.venv/bin/python scripts/analyze.py --config pilot && ./.venv/bin/python scripts/make_figures.py --config pilot
```

Full run (3 seeds x 5-member ensemble):

```bash
./.venv/bin/python scripts/run_experiment.py --config main
```

On a Kaggle GPU:

```bash
./.venv/bin/python scripts/kaggle_run.py prepare && ./.venv/bin/python scripts/kaggle_run.py push
```

The kernel embeds the source inline (no dataset mount) and installs a Pascal-compatible
torch if Kaggle allocates a P100, whose sm_60 capability modern torch wheels do not support.

## Integrity checks

Both are executable, not merely claimed:

```bash
./.venv/bin/python tests/test_leakage.py
```

- **No future leakage** — corrupting every future field leaves all features bit-identical
  while `H*` changes (6.45 -> 0.01).
- **Displacement identity** — integrating the *true* factors reproduces the true displacement
  exactly on every config; asserted at the start of every feature-extraction pass.

## Layout

```
configs/      experiment configs (pilot / main), including all seeds and splits
src/
  envs/       env configs A-D and the slip wrapper
  data/       collection policy, trajectory storage, sequence sampling
  models/     RSSM-lite world model + training
  uncertainty/three uncertainty families and the time-t feature extractor
  horizon/    E_H, H*, and the predictor g_phi
  planning/   adaptive-horizon MPC
  evaluation/ baselines, statistics, figures, analysis driver
scripts/      run_experiment, analyze, make_figures, run_planning, diagnose, survey_envs
tests/        leakage test
```

Every run records config, seeds, git commit, package versions, and GPU info to
`results/<config>/provenance.json`.
