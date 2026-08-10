"""Phase 4: uncertainty signals + the time-t feature extractor.

Three families, as pre-registered:
  A. predictive entropy / variance          -> `ent_*`
  B. ensemble disagreement (K members)      -> `js_*`, `dispvar_*`
  C. latent uncertainty                     -> `prior_std`, `post_prior_kl`

Feature groups (used for ablation and to enforce the information boundary):
  latent      member-0 belief state at the anchor
  apriori     uncertainty computable at t with a single 1-step lookahead
  imag        uncertainty read off the *imagined* rollout (no real future observed)
  actions     the planned action sequence, known to a planner at time t
  hand        hand-crafted statistics of the current observation only

NOTHING here may touch a real observation, state, reward or error at time > t.
`assert_no_future_leakage` in tests/test_leakage.py verifies this by perturbing the future.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.envs.configs import A_FORWARD, AGENT_VIEW, N_ACTIONS
from src.horizon.rih import CONTEXT, H_MAX

PROBE_HORIZONS = [1, 2, 5, 10, 20, 30]
# MiniGrid object indices
IDX_UNSEEN, IDX_EMPTY, IDX_WALL, IDX_FLOOR, IDX_DOOR, IDX_KEY = 0, 1, 2, 3, 4, 5
IDX_GOAL, IDX_LAVA = 8, 9
PASSABLE_IDS = (IDX_EMPTY, IDX_FLOOR, IDX_GOAL)


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p.clamp_min(1e-9).log() * p).sum(-1)


def _js(probs: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence across the leading (member) axis. (K,...,C) -> (...)"""
    m = probs.mean(0)
    return _entropy(m) - _entropy(probs).mean(0)


@torch.no_grad()
def extract_features(models, batch, rollout_out, context: int = CONTEXT,
                     h_max: int = H_MAX) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Build the time-t feature matrix. Returns (X, names, group -> column indices)."""
    r = rollout_out
    B = batch["dir"].shape[0]
    anchor = context - 1
    dev = batch["dir"].device
    feats: list[np.ndarray] = []
    names: list[str] = []
    groups: dict[str, list[int]] = {}

    def add(group: str, name: str, arr: torch.Tensor | np.ndarray):
        a = arr.detach().float().cpu().numpy() if torch.is_tensor(arr) else np.asarray(arr, np.float32)
        a = a.reshape(B, -1)
        groups.setdefault(group, []).extend(range(len(names), len(names) + a.shape[1]))
        names.extend([name] if a.shape[1] == 1 else [f"{name}_{i}" for i in range(a.shape[1])])
        feats.append(a.astype(np.float32))

    # ---- member probabilities per imagined step: (K,B,H,C) -------------
    pm_moved = torch.stack([F.softmax(o["moved"], -1) for o in r["per_member"]])
    pm_dir = torch.stack([F.softmax(o["dir"], -1) for o in r["per_member"]])
    p_moved, p_dir = r["p_moved"], r["p_dir"]

    ent_moved = _entropy(p_moved)              # (B,H)
    ent_dir = _entropy(p_dir)
    js_moved = _js(pm_moved)                   # (B,H)
    js_dir = _js(pm_dir)
    prior_std = torch.stack([o["prior_std"].mean(-1) for o in r["per_member"]]).mean(0)  # (B,H)

    # disagreement of imagined *displacement* across members: (K,B,H,2) -> (B,H)
    dm = r["disp_members"].float()
    dispvar = dm.var(0, unbiased=False).sum(-1)
    dispspread = (dm - dm.mean(0, keepdim=True)).abs().sum(-1).mean(0)

    # ---- A-PRIORI: single 1-step lookahead ------------------------------
    add("apriori", "ent_moved_1", ent_moved[:, 0])
    add("apriori", "ent_dir_1", ent_dir[:, 0])
    add("apriori", "js_moved_1", js_moved[:, 0])
    add("apriori", "js_dir_1", js_dir[:, 0])

    # ---- C: latent uncertainty at the anchor ----------------------------
    pstd = torch.stack([s["prior_std"].mean(-1) for s in r["anchor_state"]]).mean(0)
    qstd = torch.stack([s["post_std"].mean(-1) for s in r["anchor_state"]]).mean(0)
    kl = torch.stack([
        (torch.log(s["prior_std"] / s["post_std"])
         + (s["post_std"] ** 2 + (s["post_mu"] - s["prior_mu"]) ** 2) / (2 * s["prior_std"] ** 2)
         - 0.5).sum(-1)
        for s in r["anchor_state"]
    ]).mean(0)
    add("apriori", "prior_std_anchor", pstd)
    add("apriori", "post_std_anchor", qstd)
    add("apriori", "post_prior_kl", kl)

    # one-step-ahead tracking error over the observed context (Baseline 3)
    for key in ("ctx_moved_err", "ctx_dir_err", "ctx_moved_err_last"):
        add("apriori", key, torch.stack([s[key] for s in r["anchor_state"]]).mean(0))

    # ---- IMAGINATION-INFORMED (no real future observed) -----------------
    for h in PROBE_HORIZONS:
        i = h - 1
        add("imag", f"js_moved_h{h}", js_moved[:, i])
        add("imag", f"ent_moved_h{h}", ent_moved[:, i])
        add("imag", f"dispvar_h{h}", dispvar[:, i])
        add("imag", f"dispspread_h{h}", dispspread[:, i])
        add("imag", f"prior_std_h{h}", prior_std[:, i])
    add("imag", "js_moved_mean", js_moved.mean(1))
    add("imag", "js_moved_cumsum", js_moved.cumsum(1)[:, -1])
    add("imag", "ent_moved_mean", ent_moved.mean(1))
    add("imag", "dispspread_mean", dispspread.mean(1))
    # First horizon at which the imagined rollouts diverge past a threshold. This is the
    # classical "online truncation" signal (MBPO/MACURA-style) expressed as a horizon, and
    # is therefore an important baseline rather than a mere feature.
    add("imag", "trunc_h_disp", (dispspread < 1.0).long().cumprod(1).sum(1))
    add("imag", "trunc_h_js", (js_moved < 0.1).long().cumprod(1).sum(1))

    # ---- LATENT state (member 0 belief) ---------------------------------
    add("latent", "h0", r["anchor_state"][0]["h"])
    add("latent", "z0", r["anchor_state"][0]["z"])

    # ---- ACTIONS: the planned sequence, known at planning time ----------
    acts = r["actions"]                                    # (B,H)
    onehot = F.one_hot(acts, N_ACTIONS).float()
    add("actions", "act_count", onehot.sum(1))
    is_fwd = (acts == A_FORWARD).float()
    add("actions", "n_forward", is_fwd.sum(1))
    add("actions", "fwd_cumsum", is_fwd.cumsum(1)[:, [h - 1 for h in PROBE_HORIZONS]])
    add("actions", "first_fwd", torch.where(
        is_fwd.any(1), is_fwd.argmax(1).float(), torch.full((B,), float(h_max), device=dev)))

    # ---- HAND-CRAFTED observation statistics at the anchor --------------
    img = batch["image"][:, anchor]                        # (B,V,V,3)
    obj = img[..., 0]
    for idx, nm in [(IDX_UNSEEN, "unseen"), (IDX_WALL, "wall"), (IDX_DOOR, "door"),
                    (IDX_KEY, "key"), (IDX_EMPTY, "empty"), (IDX_LAVA, "lava")]:
        add("hand", f"n_{nm}", (obj == idx).float().sum((-1, -2)))
    # agent sits at (x=V//2, y=V-1) in view coords, facing decreasing y
    cx, cy = AGENT_VIEW // 2, AGENT_VIEW - 1
    ahead = obj[:, cx, :cy].flip(-1)                       # (B, V-1) nearest cell first
    passable = torch.zeros_like(ahead, dtype=torch.bool)
    for pid in PASSABLE_IDS:
        passable |= ahead == pid
    add("hand", "free_ahead", passable.long().cumprod(1).sum(1).float())
    add("hand", "unseen_ahead", (ahead == IDX_UNSEEN).float().sum(1))
    add("hand", "carrying_anchor", batch["carrying"][:, anchor].float())
    add("hand", "door_open_anchor", batch["door_open"][:, anchor].float())
    add("hand", "dir_anchor", F.one_hot(batch["dir"][:, anchor], 4).float())

    X = np.concatenate(feats, axis=1)
    assert X.shape[1] == len(names)
    return X, names, groups


SCALAR_SIGNALS = {
    # single-scalar uncertainty signals used for the isotonic Claim-A ceiling and baselines
    "ensemble_disagreement_1step": "js_moved_1",
    "predictive_entropy_1step": "ent_moved_1",
    "latent_prior_std": "prior_std_anchor",
    "latent_post_prior_kl": "post_prior_kl",
    "ensemble_disagreement_mean": "js_moved_mean",
    "imagined_disp_spread": "dispspread_mean",
    "internal_truncation_disp": "trunc_h_disp",
    "internal_truncation_js": "trunc_h_js",
    "free_ahead": "free_ahead",
    "n_forward_planned": "n_forward",
    "context_onestep_error": "ctx_moved_err",
}
