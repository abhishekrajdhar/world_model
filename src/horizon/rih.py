"""Phases 5-6: imagination rollout, error curve E_H, and ground-truth RIH H*.

Definitions are frozen in PREREGISTRATION.md:
    E_H     = Manhattan distance between predicted and true displacement after H steps
    epsilon = 1  (integer displacements, so E_H < 1 means an exact cell match)
    H*      = max{H <= H_max : E_h = 0 for all h in 1..H}   (first-failure definition)

The agent's world model is the K-member ensemble; its point prediction at each imagined step
is the argmax of the members' averaged head distributions. Member 0 alone is also evaluated
as a robustness check.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.models.world_model import DIR_VEC, integrate_displacement

CONTEXT = 16   # observed steps used to form the belief at the anchor (frozen)
H_MAX = 30     # maximum imagination horizon (frozen)
EPSILON = 1    # failure threshold on Manhattan displacement error (frozen)


@torch.no_grad()
def rollout(models, batch, context: int = CONTEXT, h_max: int = H_MAX,
            actions: torch.Tensor | None = None, need_truth: bool = True):
    """Filter to the anchor, then imagine `h_max` steps under an action sequence.

    `actions` defaults to the real actions in `batch`. During planning the caller supplies
    candidate sequences instead and sets `need_truth=False`, since the future is unavailable.
    Returns per-member and ensemble-consensus predictions (and ground truth if requested).
    Shapes: K members, B anchors, H horizons.
    """
    anchor = context - 1
    if actions is None:
        actions = batch["action"][:, anchor: anchor + h_max]        # (B,H) actions at t..t+H-1
    dir0 = batch["dir"][:, anchor]                                   # (B,)

    per_member = []
    anchor_state = []
    for m in models:
        h, z, aux = m.filter(batch, upto=context)
        out = m.imagine(h, z, actions, sample=False)
        per_member.append(out)
        # `h` is the recurrent state for the GRU model, but architectures without a recurrent
        # vector (e.g. the Transformer) hand back an opaque carrier plus an explicit
        # `h_repr` tensor to use as the belief representation.
        h_repr = aux.pop("h_repr", h)
        anchor_state.append({"h": h_repr, "z": z, **aux})

    # ensemble-averaged head probabilities at each imagined step
    p_dir = torch.stack([F.softmax(o["dir"], -1) for o in per_member]).mean(0)      # (B,H,4)
    p_moved = torch.stack([F.softmax(o["moved"], -1) for o in per_member]).mean(0)  # (B,H,2)

    ens_dir = p_dir.argmax(-1)
    ens_moved = p_moved.argmax(-1)
    disp_ens = integrate_displacement(ens_dir, ens_moved, dir0)                      # (B,H,2)

    disp_members = torch.stack([
        integrate_displacement(o["dir"].argmax(-1), o["moved"].argmax(-1), dir0)
        for o in per_member
    ])                                                                               # (K,B,H,2)

    if need_truth:
        true_pos = batch["pos"][:, anchor: anchor + h_max + 1]                       # (B,H+1,2)
        disp_true = true_pos[:, 1:] - true_pos[:, :1]                                # (B,H,2)
    else:
        disp_true = torch.zeros_like(disp_ens)

    return {
        "disp_ens": disp_ens,
        "disp_members": disp_members,
        "disp_true": disp_true,
        "p_dir": p_dir,
        "p_moved": p_moved,
        "per_member": per_member,
        "anchor_state": anchor_state,
        "actions": actions,
        "dir0": dir0,
    }


def error_curve(disp_pred: torch.Tensor, disp_true: torch.Tensor) -> torch.Tensor:
    """E_H for H = 1..H_max. (B,H,2) -> (B,H) Manhattan error."""
    return (disp_pred - disp_true).abs().sum(-1)


def rih_from_errors(E: torch.Tensor, epsilon: int = EPSILON) -> torch.Tensor:
    """First-failure RIH: number of leading horizons with E_h < epsilon.

    E: (B,H). Returns (B,) integer in [0, H].
    """
    ok = (E < epsilon).long()                       # (B,H)
    # cumulative product of `ok` = 1 while no failure has occurred yet
    return ok.cumprod(dim=1).sum(dim=1)


def rih_nonmonotone(E: torch.Tensor, epsilon: int = EPSILON) -> torch.Tensor:
    """Robustness variant: largest H with E_H < epsilon, ignoring earlier failures."""
    B, H = E.shape
    ok = E < epsilon
    idx = torch.arange(1, H + 1, device=E.device).expand(B, H)
    return torch.where(ok, idx, torch.zeros_like(idx)).max(dim=1).values


def compute_ground_truth(models, batch, context: int = CONTEXT, h_max: int = H_MAX,
                         epsilon: int = EPSILON) -> dict:
    r = rollout(models, batch, context, h_max)
    E_ens = error_curve(r["disp_ens"], r["disp_true"])
    E_m0 = error_curve(r["disp_members"][0], r["disp_true"])
    return {
        "E_ens": E_ens,
        "E_m0": E_m0,
        "H_star": rih_from_errors(E_ens, epsilon),
        "H_star_nonmono": rih_nonmonotone(E_ens, epsilon),
        "H_star_m0": rih_from_errors(E_m0, epsilon),
        "rollout": r,
    }


def true_displacement_check(batch, context: int = CONTEXT, h_max: int = H_MAX) -> bool:
    """Sanity: integrating the TRUE dir/moved factors must reproduce the true displacement.

    This validates the factorised state representation end to end. If it fails, the
    ground-truth target is wrong and every downstream number is meaningless.
    """
    anchor = context - 1
    dir_seq = batch["dir"][:, anchor + 1: anchor + h_max + 1]
    moved_seq = batch["moved"][:, anchor + 1: anchor + h_max + 1]
    dir0 = batch["dir"][:, anchor]
    disp = integrate_displacement(dir_seq, moved_seq, dir0)
    true_pos = batch["pos"][:, anchor: anchor + h_max + 1]
    disp_true = true_pos[:, 1:] - true_pos[:, :1]
    return bool((disp == disp_true).all())
