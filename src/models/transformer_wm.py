"""A deliberately DIFFERENT world-model architecture, for the architecture-generality test.

The main model is a GRU-based RSSM: gated recurrence, autoregressive latent feedback.
This one shares none of that:

  * no recurrence at all -- causal self-attention aggregates the history
  * no autoregressive latent feedback -- the latent at t is a function of the attended
    history, not of z_{t-1}, so both training and imagination are single parallel passes
  * observations are replaced by a learned MASK token wherever they are unavailable, which
    is how imagination is expressed instead of "roll the prior forward"

It exposes exactly the same public API (`observe`, `loss`, `filter`, `imagine`) and produces
exactly the same head set, so `src/horizon/rih.py` and `src/uncertainty/signals.py` operate on
it unchanged. That is what makes the cross-architecture comparison meaningful: identical
targets, identical features, identical downstream code -- only the dynamics differ.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.configs import N_ACTIONS
from src.models.world_model import N_DIR, ObsEncoder, WMConfig, _kl


@dataclass
class TransformerWMConfig(WMConfig):
    d_model: int = 256
    n_layers: int = 3
    n_heads: int = 4
    max_len: int = 64
    # WMConfig fields kept so the shared ObsEncoder and feature code work unchanged
    stoch: int = 32


class TransformerWorldModel(nn.Module):
    def __init__(self, cfg: TransformerWMConfig = TransformerWMConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = ObsEncoder(cfg)
        self.action_emb = nn.Embedding(N_ACTIONS, 32)
        self.mask_token = nn.Parameter(torch.zeros(cfg.obs_embed))
        nn.init.normal_(self.mask_token, std=0.02)
        self.in_proj = nn.Linear(cfg.obs_embed + 32, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=4 * cfg.d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

        self.latent = nn.Linear(cfg.d_model, 2 * cfg.stoch)
        self.trunk = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.stoch, cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU())
        self.head_dir = nn.Linear(cfg.hidden, N_DIR)
        self.head_moved = nn.Linear(cfg.hidden, 2)
        self.head_carry = nn.Linear(cfg.hidden, 2)
        self.head_door = nn.Linear(cfg.hidden, 2)

    # ---------- core ----------
    def _features(self, batch, obs_mask: torch.Tensor) -> torch.Tensor:
        """obs_mask (B,L) bool: True where the observation at t may be used."""
        B, L = batch["dir"].shape
        e = self.encoder(batch["image"], batch["dir"], batch["carrying"].float())  # (B,L,E)
        e = torch.where(obs_mask[..., None], e, self.mask_token.expand_as(e))
        a_prev = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=e.device),
                            batch["action"][:, :-1]], 1)
        tok = self.in_proj(torch.cat([e, self.action_emb(a_prev)], -1)) + self.pos[:, :L]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=e.device), 1)
        return self.blocks(tok, mask=causal)

    def _decode(self, feat, sample: bool):
        mu, raw = self.latent(feat).chunk(2, -1)
        std = F.softplus(raw) + 0.1
        z = mu + std * torch.randn_like(std) if sample else mu
        h = self.trunk(torch.cat([feat, z], -1))
        return {"dir": self.head_dir(h), "moved": self.head_moved(h),
                "carrying": self.head_carry(h), "door_open": self.head_door(h)}, mu, std

    # ---------- training ----------
    def observe(self, batch, split_point: int, sample: bool = True):
        B, L = batch["dir"].shape
        dev = batch["dir"].device
        idx = torch.arange(L, device=dev)[None, :].expand(B, L)
        post_mask = idx < split_point                      # observations available
        feat_post = self._features(batch, post_mask)
        out, qm, qs = self._decode(feat_post, sample)
        # "prior" = same model with the observation at each step withheld
        feat_prior = self._features(batch, torch.zeros_like(post_mask))
        _, pm, ps = self._decode(feat_prior, sample=False)
        out["kl"] = _kl(qm[:, :split_point], qs[:, :split_point],
                        pm[:, :split_point], ps[:, :split_point])
        return out

    def loss(self, batch, split_point: int):
        out = self.observe(batch, split_point)
        targets = {"dir": batch["dir"], "moved": batch["moved"].long(),
                   "carrying": batch["carrying"].long(), "door_open": batch["door_open"].long()}
        weights = {"dir": 1.0, "moved": 2.0, "carrying": 0.3, "door_open": 0.3}
        ce, parts = 0.0, {}
        for k, tgt in targets.items():
            l = F.cross_entropy(out[k].reshape(-1, out[k].shape[-1]), tgt.reshape(-1))
            parts[k] = l.detach()
            ce = ce + weights[k] * l
        kl = out["kl"].mean() if out["kl"].numel() else torch.zeros((), device=ce.device)
        total = ce + self.cfg.kl_scale * torch.clamp(kl, min=self.cfg.free_nats)
        parts["kl"], parts["total"] = kl.detach(), total.detach()
        return total, parts

    # ---------- inference ----------
    @torch.no_grad()
    def filter(self, batch, upto: int):
        """Returns (state, z, aux) with the same aux keys as the GRU model.

        `state` carries the observed prefix so `imagine` can continue from it; there is no
        recurrent hidden vector to hand over.
        """
        B = batch["dir"].shape[0]
        dev = batch["dir"].device
        sub = {k: v[:, :upto] for k, v in batch.items()}
        idx = torch.arange(upto, device=dev)[None, :].expand(B, upto)

        feat_post = self._features(sub, torch.ones_like(idx, dtype=torch.bool))
        _, qm, qs = self._decode(feat_post, sample=False)
        feat_prior = self._features(sub, idx < upto - 1)      # withhold only the anchor obs
        _, pm, ps = self._decode(feat_prior, sample=False)

        # One-step-ahead tracking error over the context (Baseline 3).
        # NOTE a definitional difference from the GRU model: there, the prior at t sees all
        # observations before t. Here a per-token "everything strictly before t" mask is not
        # expressible in a single pass, so we use the fully-masked pass, which withholds MORE
        # information. This makes the feature strictly harder for the Transformer, so any
        # cross-architecture advantage it shows cannot be an artefact of this choice.
        moved_err = torch.zeros(B, 1, device=dev)
        dir_err = torch.zeros(B, 1, device=dev)
        if upto > 1:
            pred, _, _ = self._decode(feat_prior, sample=False)
            moved_err = (pred["moved"].argmax(-1)[:, 1:] != sub["moved"][:, 1:]).float()
            dir_err = (pred["dir"].argmax(-1)[:, 1:] != sub["dir"][:, 1:]).float()
        aux = {
            "prior_mu": pm[:, -1], "prior_std": ps[:, -1],
            "post_mu": qm[:, -1], "post_std": qs[:, -1],
            "ctx_moved_err": moved_err.mean(1),
            "ctx_dir_err": dir_err.mean(1),
            "ctx_moved_err_last": moved_err[:, -1],
        }
        return {"batch": sub, "upto": upto}, qm[:, -1], aux

    @torch.no_grad()
    def imagine(self, state, z, actions, sample: bool = False):
        """One parallel pass: context tokens keep their observations, imagined steps are
        MASK tokens conditioned on the given actions."""
        sub, upto = state["batch"], state["upto"]
        B, H = actions.shape
        dev = actions.device
        L = upto + H
        pad = {}
        for k, v in sub.items():
            rep = v[:, -1:].expand(-1, H, *([-1] * (v.dim() - 2)))
            pad[k] = torch.cat([v, rep], 1)
        # actions[:, j] is taken AT step upto-1+j, i.e. it precedes imagined step upto+j.
        # The token sequence has length L = upto + H, and _features reads only action[:-1]
        # (as a_prev), so the final slot is never used and is padded to keep lengths aligned.
        pad["action"] = torch.cat(
            [sub["action"][:, :upto - 1], actions,
             torch.zeros(B, 1, dtype=actions.dtype, device=dev)], 1)
        assert pad["action"].shape[1] == L, (pad["action"].shape, L)
        idx = torch.arange(L, device=dev)[None, :].expand(B, L)
        feat = self._features(pad, idx < upto)
        out, _, std = self._decode(feat, sample=sample)
        out = {k: v[:, upto:] for k, v in out.items()}
        out["prior_std"] = std[:, upto:]
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
