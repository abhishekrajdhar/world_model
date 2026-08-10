"""Phase 3: latent world model (RSSM-lite).

    obs -> encoder -> e_t
    (h_{t-1}, z_{t-1}, a_{t-1}) -> GRU -> h_t
    prior      p(z_t | h_t)              <- used during imagination (no observation)
    posterior  q(z_t | h_t, e_t)         <- used while observations are available
    [h_t, z_t] -> heads -> dir, moved, carrying, door_open

Decoder heads are size-agnostic state *factors*. Absolute displacement is reconstructed by
integrating `moved` along `dir`, so position error is a deterministic function of the heads
(see PREREGISTRATION.md sec. 5-6).

Training deliberately mixes filtering and open-loop imagination: for each sequence a split
point c is drawn, the posterior is used for t < c and the prior alone for t >= c. This
matches the quantity we evaluate (open-loop imagination) rather than only one-step accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.configs import AGENT_VIEW, N_ACTIONS

N_OBJ, N_COLOR, N_STATE = 11, 6, 3
N_DIR = 4
# direction vectors, matching MiniGrid: 0=right, 1=down, 2=left, 3=up
DIR_VEC = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=torch.long)


@dataclass
class WMConfig:
    deter: int = 512          # GRU hidden size h
    stoch: int = 48           # latent z size
    hidden: int = 512         # MLP width
    obs_embed: int = 384
    obj_emb: int = 16
    col_emb: int = 6
    st_emb: int = 6
    free_nats: float = 1.0
    kl_scale: float = 1.0


class ObsEncoder(nn.Module):
    def __init__(self, cfg: WMConfig):
        super().__init__()
        self.obj = nn.Embedding(N_OBJ, cfg.obj_emb)
        self.col = nn.Embedding(N_COLOR, cfg.col_emb)
        self.st = nn.Embedding(N_STATE, cfg.st_emb)
        per_cell = cfg.obj_emb + cfg.col_emb + cfg.st_emb
        flat = per_cell * AGENT_VIEW * AGENT_VIEW + N_DIR + 1
        self.net = nn.Sequential(
            nn.Linear(flat, cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.obs_embed), nn.SiLU(),
        )

    def forward(self, image, direction, carrying):
        # image (..., V, V, 3) long; direction (...,) long; carrying (...,) float
        e = torch.cat([self.obj(image[..., 0]), self.col(image[..., 1]), self.st(image[..., 2])], -1)
        e = e.flatten(-3)
        e = torch.cat([e, F.one_hot(direction, N_DIR).float(), carrying[..., None].float()], -1)
        return self.net(e)


class WorldModel(nn.Module):
    def __init__(self, cfg: WMConfig = WMConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = ObsEncoder(cfg)
        self.action_emb = nn.Embedding(N_ACTIONS, 32)
        self.cell = nn.GRUCell(cfg.stoch + 32, cfg.deter)

        self.prior_net = nn.Sequential(
            nn.Linear(cfg.deter, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, 2 * cfg.stoch)
        )
        self.post_net = nn.Sequential(
            nn.Linear(cfg.deter + cfg.obs_embed, cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, 2 * cfg.stoch),
        )
        feat = cfg.deter + cfg.stoch
        self.trunk = nn.Sequential(nn.Linear(feat, cfg.hidden), nn.SiLU(),
                                   nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU())
        self.head_dir = nn.Linear(cfg.hidden, N_DIR)
        self.head_moved = nn.Linear(cfg.hidden, 2)
        self.head_carry = nn.Linear(cfg.hidden, 2)
        self.head_door = nn.Linear(cfg.hidden, 2)

    # --- distributions -------------------------------------------------
    @staticmethod
    def _split(params):
        mu, raw = params.chunk(2, -1)
        std = F.softplus(raw) + 0.1
        return mu, std

    def prior(self, h):
        return self._split(self.prior_net(h))

    def posterior(self, h, e):
        return self._split(self.post_net(torch.cat([h, e], -1)))

    def heads(self, h, z):
        f = self.trunk(torch.cat([h, z], -1))
        return {
            "dir": self.head_dir(f),
            "moved": self.head_moved(f),
            "carrying": self.head_carry(f),
            "door_open": self.head_door(f),
        }

    def init_state(self, batch, device):
        return (torch.zeros(batch, self.cfg.deter, device=device),
                torch.zeros(batch, self.cfg.stoch, device=device))

    def step(self, h, z, a, e=None, sample=True):
        """One recurrent step. Uses the posterior when `e` is given, else the prior."""
        h = self.cell(torch.cat([z, self.action_emb(a)], -1), h)
        pm, ps = self.prior(h)
        if e is None:
            mu, std = pm, ps
        else:
            mu, std = self.posterior(h, e)
        z = mu + std * torch.randn_like(std) if sample else mu
        return h, z, (pm, ps), (mu, std)

    # --- training ------------------------------------------------------
    def observe(self, batch, split_point: int, sample: bool = True):
        """Roll a sequence: posterior for t < split_point, prior-only afterwards.

        batch keys: image (B,L,V,V,3), dir (B,L), carrying (B,L), action (B,L),
                    moved (B,L), door_open (B,L)
        Returns per-step head logits and KL terms.
        """
        B, L = batch["dir"].shape
        device = batch["dir"].device
        e_all = self.encoder(batch["image"], batch["dir"], batch["carrying"].float())

        h, z = self.init_state(B, device)
        logits, kls = [], []
        for t in range(L):
            a_prev = batch["action"][:, t - 1] if t > 0 else torch.zeros(B, dtype=torch.long, device=device)
            e = e_all[:, t] if t < split_point else None
            h, z, (pm, ps), (qm, qs) = self.step(h, z, a_prev, e, sample=sample)
            logits.append(self.heads(h, z))
            if t < split_point:
                kls.append(_kl(qm, qs, pm, ps))
        out = {k: torch.stack([lg[k] for lg in logits], 1) for k in logits[0]}
        out["kl"] = torch.stack(kls, 1) if kls else torch.zeros(B, 0, device=device)
        return out

    def loss(self, batch, split_point: int):
        out = self.observe(batch, split_point)
        L = batch["dir"].shape[1]
        ce = 0.0
        targets = {"dir": batch["dir"], "moved": batch["moved"].long(),
                   "carrying": batch["carrying"].long(), "door_open": batch["door_open"].long()}
        # weight `moved` and `dir` highest: they define the primary error metric
        weights = {"dir": 1.0, "moved": 2.0, "carrying": 0.3, "door_open": 0.3}
        parts = {}
        for k, tgt in targets.items():
            l = F.cross_entropy(out[k].reshape(-1, out[k].shape[-1]), tgt.reshape(-1))
            parts[k] = l.detach()
            ce = ce + weights[k] * l
        kl = out["kl"].mean() if out["kl"].numel() else torch.zeros((), device=ce.device)
        kl_clipped = torch.clamp(kl, min=self.cfg.free_nats)
        total = ce + self.cfg.kl_scale * kl_clipped
        parts["kl"] = kl.detach()
        parts["total"] = total.detach()
        return total, parts

    # --- inference / imagination ---------------------------------------
    @torch.no_grad()
    def filter(self, batch, upto: int):
        """Run the posterior over t = 0..upto-1, return the belief state at t = upto-1.

        Also returns the model's *one-step-ahead* prediction error over the context: at each
        step the prior (which has not yet seen o_t) decodes a prediction of the state, which
        is scored against the observed state. This is available online at time t and forms
        Baseline 3 ("prediction error at the current step").
        """
        B = batch["dir"].shape[0]
        device = batch["dir"].device
        e_all = self.encoder(batch["image"][:, :upto], batch["dir"][:, :upto],
                             batch["carrying"][:, :upto].float())
        h, z = self.init_state(B, device)
        aux = {}
        moved_err, dir_err = [], []
        for t in range(upto):
            a_prev = batch["action"][:, t - 1] if t > 0 else torch.zeros(B, dtype=torch.long, device=device)
            h_new = self.cell(torch.cat([z, self.action_emb(a_prev)], -1), h)
            pm, ps = self.prior(h_new)
            if t > 0:  # one-step-ahead prediction from the prior, before seeing o_t
                pred = self.heads(h_new, pm)
                moved_err.append((pred["moved"].argmax(-1) != batch["moved"][:, t]).float())
                dir_err.append((pred["dir"].argmax(-1) != batch["dir"][:, t]).float())
            qm, qs = self.posterior(h_new, e_all[:, t])
            h, z = h_new, qm
            if t == upto - 1:
                aux = {"prior_mu": pm, "prior_std": ps, "post_mu": qm, "post_std": qs}
        aux["ctx_moved_err"] = torch.stack(moved_err, 1).mean(1) if moved_err else torch.zeros(B, device=device)
        aux["ctx_dir_err"] = torch.stack(dir_err, 1).mean(1) if dir_err else torch.zeros(B, device=device)
        aux["ctx_moved_err_last"] = moved_err[-1] if moved_err else torch.zeros(B, device=device)
        return h, z, aux

    @torch.no_grad()
    def imagine(self, h, z, actions, sample: bool = False):
        """Open-loop rollout. `actions` (B,H) are the actions taken AT steps t..t+H-1.

        Returns dict of head logits with shape (B,H,·) for imagined steps t+1..t+H.
        """
        outs = []
        H = actions.shape[1]
        for k in range(H):
            h, z, (pm, ps), _ = self.step(h, z, actions[:, k], e=None, sample=sample)
            hd = self.heads(h, z)
            hd["prior_std"] = ps
            outs.append(hd)
        return {k: torch.stack([o[k] for o in outs], 1) for k in outs[0]}


def _kl(qm, qs, pm, ps):
    return (torch.log(ps / qs) + (qs**2 + (qm - pm) ** 2) / (2 * ps**2) - 0.5).sum(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def integrate_displacement(dir_seq: torch.Tensor, moved_seq: torch.Tensor,
                           dir0: torch.Tensor) -> torch.Tensor:
    """Displacement after H steps from `moved` flags and the direction held *before* each move.

    dir_seq   (B,H) predicted absolute direction at steps t+1..t+H
    moved_seq (B,H) whether the transition into t+k changed the cell
    dir0      (B,)  direction at the anchor t
    Returns cumulative displacement (B,H,2).
    """
    B, H = dir_seq.shape
    prev_dir = torch.cat([dir0[:, None], dir_seq[:, :-1]], 1)  # direction held before each move
    vec = DIR_VEC.to(dir_seq.device)[prev_dir]                 # (B,H,2)
    step = vec * moved_seq[..., None].long()
    return step.cumsum(1)
