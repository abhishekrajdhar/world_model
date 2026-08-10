"""Phase 7: the lightweight horizon predictor g_phi(z_t, U_t, a_t:t+H) -> H*.

Outputs a full categorical distribution over H* in {0..H_MAX}, which gives both a point
estimate and the P(H* >= h) curve needed for the calibration analysis.

Point estimate convention (frozen): the conditional MEAN of the predictive distribution is
used for MAE/RMSE/correlations. The same convention is applied to every baseline (isotonic
regression also returns a conditional mean), so comparisons are like-for-like.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.horizon.rih import H_MAX
from src.utils.seeding import set_seed


class HorizonPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, n_classes: int = H_MAX + 1, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, n_classes)]
        self.net = nn.Sequential(*layers)
        self.n_classes = n_classes

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict(self, x) -> tuple[np.ndarray, np.ndarray]:
        p = F.softmax(self(x), -1)
        vals = torch.arange(self.n_classes, device=x.device).float()
        mean = (p * vals).sum(-1)
        return mean.cpu().numpy(), p.cpu().numpy()


class Standardizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mu = X.mean(0, keepdims=True)
        self.sd = X.std(0, keepdims=True) + 1e-6
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mu) / self.sd, -10, 10).astype(np.float32)


def train_predictor(
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
    seed: int = 0, epochs: int = 60, batch_size: int = 512, lr: float = 1e-3,
    hidden: int = 256, depth: int = 2, device: str | torch.device = "cpu",
    weight_decay: float = 1e-4, verbose: bool = False,
) -> tuple[HorizonPredictor, Standardizer, dict]:
    set_seed(seed)
    device = torch.device(device)
    std = Standardizer().fit(X_tr)
    xt = torch.as_tensor(std(X_tr), device=device)
    yt = torch.as_tensor(y_tr.astype(np.int64), device=device)
    xv = torch.as_tensor(std(X_va), device=device)
    yv = torch.as_tensor(y_va.astype(np.int64), device=device)

    model = HorizonPredictor(X_tr.shape[1], hidden, H_MAX + 1, depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(xt)
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    hist = {"epoch": [], "val_mae": [], "val_nll": []}

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = torch.as_tensor(perm[i: i + batch_size], device=device)
            loss = F.cross_entropy(model(xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mean, _ = model.predict(xv)
            nll = F.cross_entropy(model(xv), yv).item()
        mae = float(np.abs(mean - y_va).mean())
        hist["epoch"].append(ep); hist["val_mae"].append(mae); hist["val_nll"].append(nll)
        # model selection on VALIDATION MAE only (test set never consulted)
        if mae < best[0]:
            best = (mae, {k: v.detach().clone() for k, v in model.state_dict().items()})
        if verbose and (ep + 1) % 20 == 0:
            print(f"    predictor ep {ep+1}/{epochs} val_mae {mae:.3f} nll {nll:.3f}", flush=True)

    if best[1] is not None:
        model.load_state_dict(best[1])
    model.eval()
    return model, std, hist
