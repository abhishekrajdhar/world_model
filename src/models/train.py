"""World-model training loop (single ensemble member)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.data.collect import Trajectories
from src.data.dataset import SequenceSampler
from src.models.world_model import WMConfig, WorldModel, count_params
from src.utils.seeding import set_seed


def train_member(
    traj_train: Trajectories,
    traj_val: Trajectories,
    seed: int,
    steps: int = 4000,
    batch_size: int = 64,
    seq_len: int = 50,
    min_context: int = 5,
    max_context: int = 20,
    lr: float = 3e-4,
    device: torch.device | str = "cpu",
    amp: bool = False,
    wm_cfg: WMConfig | None = None,
    log_every: int = 500,
    ckpt_path: str | Path | None = None,
    verbose: bool = True,
    model: torch.nn.Module | None = None,
) -> tuple[WorldModel, dict]:
    """`model` lets a caller supply an alternative architecture; it only has to implement
    `loss`, `observe` and the inference API (see src/models/transformer_wm.py)."""
    set_seed(seed)
    device = torch.device(device)
    if model is None:
        model = WorldModel(wm_cfg or WMConfig()).to(device)
    else:
        model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sampler = SequenceSampler(traj_train, seq_len, batch_size, seed=seed)
    val_sampler = SequenceSampler(traj_val, seq_len, batch_size, seed=seed + 9999)
    rng = np.random.default_rng(seed)

    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"step": [], "train_loss": [], "val_loss": [], "val_moved_acc": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = sampler.sample(device)
        # random split point: filtering before, open-loop imagination after
        c = int(rng.integers(min_context, max_context + 1))
        model.train()
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, parts = model.loss(batch, c)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        scaler.step(opt)
        scaler.update()

        if step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vb = val_sampler.sample(device)
                vloss, vparts = model.loss(vb, min_context)
                out = model.observe(vb, min_context, sample=False)
                pred_moved = out["moved"].argmax(-1)[:, min_context:]
                acc = (pred_moved == vb["moved"][:, min_context:]).float().mean().item()
            history["step"].append(step)
            history["train_loss"].append(float(parts["total"]))
            history["val_loss"].append(float(vloss))
            history["val_moved_acc"].append(acc)
            if verbose:
                print(f"  [seed {seed}] step {step}/{steps} train {float(parts['total']):.4f} "
                      f"val {float(vloss):.4f} moved_acc(open-loop) {acc:.3f} "
                      f"kl {float(parts['kl']):.3f} ({time.time()-t0:.0f}s)", flush=True)

    if ckpt_path is not None:
        p = Path(ckpt_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "cfg": model.cfg.__dict__,
                    "seed": seed,
                    "params": count_params(model),
                    "history": history}, p)
    return model, history


def load_member(path: str | Path, device="cpu") -> WorldModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = WorldModel(WMConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model
