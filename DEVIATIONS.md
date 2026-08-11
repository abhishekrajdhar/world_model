# Deviations from the Pre-Registration

Every departure from `PREREGISTRATION.md`, with the reason and whether it occurred before or
after any test data was examined.

---

### D1 — Environment configs A-D changed (BEFORE any test evaluation)

**Original:** A = `DoorKey-8x8`, B = `MultiRoom-N4-S5`, C = A + slip, D = `DoorKey-16x16`.
**Changed to:** A = `DoorKey-16x16`, B = `FourRooms`, C = A + slip, D = `MultiRoom-N6`.

**Reason.** Phase 1 validation (`scripts/survey_envs.py`, no neural network involved) showed
the original configs put the agent in a regime where the reliable horizon is degenerate: the
agent travelled ~2.7 cells in 30 steps while seeing 4 cells ahead, so the planned path never
left the visible region. The new configs give ~5.9 cells of travel, which is the regime where
the scientific question is non-trivial.

**Status.** Diagnosed from a policy/geometry survey and from validation data only. No test
split was consulted.

---

### D2 — Data-collection policy changed (BEFORE any test evaluation)

**Original:** BFS-based scripted explorer with epsilon-greedy noise.
**Changed to:** forward-biased random walk (`p_forward = 0.6`) with occasional scripted
key/door subgoals (`p_script = 0.35`).

**Reason.** The BFS explorer produced `P(moved | forward) = 0.985` — it never walked into
walls, because BFS paths are collision-free by construction. That removed the only source of
epistemic uncertainty the experiment is designed to measure, and a "no-walls" model that
assumes every forward move succeeds scored `H* = 27`. Under the new policy
`P(moved | forward)` is 0.48-0.67 across configs.

**Status.** Diagnosed on pilot data with pilot-only test seeds (see D4). No main test split
was consulted.

---

### D3 — Splits restructured (BEFORE any data was collected)

**Original:** wm-train / val / test.
**Changed to:** wm-train / **g-train** / val / test, where `g_phi` is trained on episodes
held out from world-model training.

**Reason.** Fitting `g_phi` on episodes the world model was trained on would make `H*` there
inflated by memorisation and unrepresentative of deployment.

---

### D4 — Pilot test seeds moved outside the main test range (BEFORE main evaluation)

Pilot `test_seeds` changed from `(4800, 4860)` to `(5400, 5500)`, which is disjoint from the
main test range `(4800, 5400)`.

**Reason.** One early pilot diagnostic (`scripts/diagnose.py`) reported aggregate `dir`/`moved`
accuracies on the pilot's `A_test`, which at the time overlapped the main test range. Moving
the pilot range guarantees the main test set was never inspected. That early diagnostic
informed no threshold, and the models and data it used were discarded.

---

### D5 — Two additions to the analysis (BEFORE any test evaluation)

1. **Privileged-state comparison (Baseline 8b).** A predictor given the true local 11x11 grid
   plus the ordered action sequence. Added because the pilot could not otherwise distinguish
   "the agent cannot know this" from "`g_phi` is weak". Recorded as an empirical comparison,
   not a proven ceiling.
2. **Primary-variant selection on validation MAE.** The pre-registration named `g_phi`
   without specifying which feature-group variant is primary. The pilot showed the
   highest-dimensional variant can overfit, so the primary is now chosen by **A-val** MAE.
   Test data is never used for this choice, and all variants are reported.

---

### D6 — Planning task made non-myopic (BEFORE the reported planning result)

Planning targets are now required to need a genuine detour (shortest-path distance exceeding
Manhattan distance by >= 4 cells), and candidate action sequences are sampled from the same
forward-biased distribution used to collect training data.

**Reason.** With nearby targets and uniform action sampling, the task was solvable greedily
(`H = 3` was best), so no horizon trade-off existed and Phase 11 could not test anything.
Uniform sampling also pushed the world model off the distribution `g_phi` was fitted on.
Adaptive planning is additionally **charged** for the probe rollout it uses to estimate its
own horizon (`K * H_max` imagined states per decision); not charging it would flatter the
adaptive arm.

---

### D7 — Compute platform (operational, no scientific effect)

**Every reported number comes from Kaggle.** A local Apple Silicon (MPS) run was briefly
started as a hedge while the Kaggle kernel was pending; it was terminated before producing
any result and its partial artefacts were deleted, so nothing in this repository derives
from it.

Kaggle allocated a **Tesla P100 (sm_60)** rather than the requested T4; its preinstalled
torch (2.10+cu128) ships sm_70+ kernels only, so every CUDA launch failed with
`cudaErrorNoKernelImageForDevice`. The kernel now installs a Pascal-compatible torch
(2.5.1+cu121) when capability < 7.0 and re-executes, and `get_device()` verifies CUDA with a
real kernel launch before using it. The accelerator type cannot be selected through the
Kaggle API, only through the notebook UI. Main run: 15 world models, 4.4 h on P100.


---

### D8 — Planning re-run and its residual limitation (AFTER the main test evaluation)

The first planning attempt yielded 0/60 usable episodes (locked door => one convex room =>
no detour targets). A warm-up phase was added so the key is collected and the door opened
before the target is chosen, plus tiered detour thresholds.

**Outcome, reported as measured:** all 60 layouts became usable, but the detour requirement
still could not be met -- mean detour 0.1 cells, with 57/60 layouts falling through to the
no-detour tier. The fixed-horizon frontier is therefore monotone (success falls from 0.917 at
H=3 to 0.367 at H=30), so no interior optimum exists and H4 cannot be meaningfully tested in
this environment. This is recorded as a weak test rather than a clean refutation.

Compute note: the Kaggle GPU quota (30 h/week) was exhausted by the main run, so planning ran
on a Kaggle CPU kernel (4 h 47 m). A parallel cheaper kernel was launched as a hedge and
errored; it was not needed, and no result depends on it.
