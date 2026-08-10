"""Experiment configurations. Splits and seeds are part of the pre-registration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExpConfig:
    name: str = "main"
    # --- data (layout-seed ranges, disjoint by construction) -------------
    wm_train_seeds: tuple[int, int] = (0, 3000)      # world-model training
    g_train_seeds: tuple[int, int] = (3000, 4200)    # horizon-predictor training (WM-held-out)
    val_seeds: tuple[int, int] = (4200, 4800)        # thresholds + model selection
    test_seeds: tuple[int, int] = (4800, 5400)       # FINAL EVALUATION ONLY
    ood_seeds: tuple[int, int] = (100000, 100600)    # B/C/D
    steps_per_episode: int = 120
    policy_epsilon: float = 0.30

    # --- world model ----------------------------------------------------
    ensemble_k: int = 5
    wm_steps: int = 6000
    wm_batch: int = 64
    wm_seq_len: int = 50
    wm_lr: float = 3e-4
    deter: int = 512
    stoch: int = 48
    hidden: int = 512
    obs_embed: int = 384

    # --- evaluation -----------------------------------------------------
    anchor_stride: int = 4
    max_anchors_per_split: int = 40000
    eval_batch: int = 512

    # --- horizon predictor ----------------------------------------------
    g_epochs: int = 60
    g_hidden: int = 256
    g_depth: int = 2
    g_lr: float = 1e-3

    # --- repeats --------------------------------------------------------
    seeds: tuple[int, ...] = (0, 1, 2)
    n_boot: int = 2000


# The pilot uses test seeds OUTSIDE the main test range so that pilot diagnostics can never
# contaminate the final evaluation set.
PILOT = ExpConfig(
    name="pilot",
    wm_train_seeds=(0, 400),
    g_train_seeds=(3000, 3200),
    val_seeds=(4200, 4300),
    test_seeds=(5400, 5500),
    ood_seeds=(100000, 100100),
    steps_per_episode=120,
    ensemble_k=2,
    wm_steps=3000,
    anchor_stride=8,
    max_anchors_per_split=6000,
    g_epochs=30,
    seeds=(0,),
    n_boot=200,
)

MAIN = ExpConfig(name="main")

CONFIGS = {"pilot": PILOT, "main": MAIN}
