"""End-to-end RIH experiment driver.

    python scripts/run_experiment.py --config pilot
    python scripts/run_experiment.py --config main --seed 0

Stages are cached under results/<config>/ so an interrupted run resumes cheaply.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.experiment import CONFIGS, ExpConfig  # noqa: E402
from src.data.collect import Trajectories, collect  # noqa: E402
from src.data.dataset import episode_of, gather, window_starts  # noqa: E402
from src.envs.configs import ENV_CONFIGS  # noqa: E402
from src.horizon.rih import CONTEXT, EPSILON, H_MAX, compute_ground_truth, true_displacement_check  # noqa: E402
from src.models.train import load_member, train_member  # noqa: E402
from src.models.world_model import WMConfig, count_params  # noqa: E402
from src.uncertainty.signals import extract_features  # noqa: E402
from src.utils.seeding import get_device, save_provenance, set_seed  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- data
def build_data(cfg: ExpConfig, out_dir: Path) -> dict[str, Trajectories]:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "A_wmtrain": ("A", cfg.wm_train_seeds),
        "A_gtrain": ("A", cfg.g_train_seeds),
        "A_val": ("A", cfg.val_seeds),
        "A_test": ("A", cfg.test_seeds),
        "B_test": ("B", cfg.ood_seeds),
        "C_test": ("C", cfg.ood_seeds),
        "D_test": ("D", cfg.ood_seeds),
    }
    data = {}
    for key, (env_name, (lo, hi)) in specs.items():
        p = out_dir / f"traj_{key}.npz"
        if p.exists():
            data[key] = Trajectories.load(p)
        else:
            log(f"collecting {key} ({env_name}, seeds {lo}-{hi})")
            tr = collect(ENV_CONFIGS[env_name], list(range(lo, hi)),
                         steps_per_episode=cfg.steps_per_episode,
                         epsilon=cfg.policy_epsilon, policy_seed=lo)
            tr.save(p)
            data[key] = tr
        log(f"  {key}: {len(data[key])} episodes, {len(data[key].image)} steps")
    return data


# ------------------------------------------------------------ ensemble
def build_ensemble(cfg: ExpConfig, data, seed: int, ckpt_dir: Path, device) -> list:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wm_cfg = WMConfig(deter=cfg.deter, stoch=cfg.stoch, hidden=cfg.hidden,
                      obs_embed=cfg.obs_embed)
    models = []
    for k in range(cfg.ensemble_k):
        p = ckpt_dir / f"wm_seed{seed}_member{k}.pt"
        if p.exists():
            models.append(load_member(p, device))
            log(f"  loaded member {k} from cache")
            continue
        log(f"  training member {k}/{cfg.ensemble_k} (seed {seed*100+k})")
        m, _ = train_member(
            data["A_wmtrain"], data["A_val"], seed=seed * 100 + k,
            steps=cfg.wm_steps, batch_size=cfg.wm_batch, seq_len=cfg.wm_seq_len,
            lr=cfg.wm_lr, device=device, amp=(device.type == "cuda"),
            wm_cfg=wm_cfg, ckpt_path=p, log_every=max(1, cfg.wm_steps // 4),
        )
        m.eval()
        models.append(m)
    log(f"  ensemble ready: {count_params(models[0]):,} params/member, K={len(models)}")
    return models


# ------------------------------------------------------- feature build
@torch.no_grad()
def extract_split(models, traj: Trajectories, cfg: ExpConfig, device, tag: str) -> dict:
    L = CONTEXT + H_MAX + 1
    starts = window_starts(traj, L)
    if len(starts) == 0:
        raise RuntimeError(f"{tag}: no windows of length {L}")
    starts = starts[:: cfg.anchor_stride]
    if len(starts) > cfg.max_anchors_per_split:
        rng = np.random.default_rng(0)
        starts = np.sort(rng.choice(starts, cfg.max_anchors_per_split, replace=False))
    groups = episode_of(traj, starts)

    Xs, ys, ynm, Es, extras, privs, acts_seq = [], [], [], [], [], [], []
    names = groups_map = None
    checked = False
    for i in range(0, len(starts), cfg.eval_batch):
        b = gather(traj, starts[i: i + cfg.eval_batch], L, device)
        if not checked:
            assert true_displacement_check(b), f"{tag}: displacement integration identity failed"
            checked = True
        gt = compute_ground_truth(models, b, CONTEXT, H_MAX, EPSILON)
        X, names, groups_map = extract_features(models, b, gt["rollout"], CONTEXT, H_MAX)
        Xs.append(X)
        ys.append(gt["H_star"].cpu().numpy())
        ynm.append(gt["H_star_nonmono"].cpu().numpy())
        Es.append(gt["E_ens"].cpu().numpy().astype(np.float32))
        extras.append(gt["H_star_m0"].cpu().numpy())
        privs.append(b["priv"][:, CONTEXT - 1].cpu().numpy().astype(np.uint8))
        acts_seq.append(gt["rollout"]["actions"].cpu().numpy().astype(np.int8))

    out = {
        "X": np.concatenate(Xs), "y": np.concatenate(ys).astype(np.int64),
        "y_nonmono": np.concatenate(ynm).astype(np.int64),
        "E": np.concatenate(Es), "y_m0": np.concatenate(extras).astype(np.int64),
        "priv": np.concatenate(privs), "acts": np.concatenate(acts_seq),
        "groups": groups, "starts": starts, "names": names, "feature_groups": groups_map,
    }
    log(f"  {tag}: {len(out['y'])} anchors, H* mean {out['y'].mean():.2f} "
        f"sd {out['y'].std():.2f} iqr {np.percentile(out['y'],75)-np.percentile(out['y'],25):.1f} "
        f"ceiling_frac {(out['y']>=H_MAX).mean():.3f} zero_frac {(out['y']==0).mean():.3f}")
    return out


def add_latent_nn_distance(splits: dict, ref_key: str = "A_gtrain", n_ref: int = 4000):
    """Baseline 6 feature: distance from the anchor latent to the nearest training latent."""
    names = splits[ref_key]["names"]
    cols = [i for i, n in enumerate(names) if n.startswith("h0_")]
    if not cols:
        return
    rng = np.random.default_rng(0)
    R = splits[ref_key]["X"][:, cols]
    if len(R) > n_ref:
        R = R[rng.choice(len(R), n_ref, replace=False)]
    Rt = torch.as_tensor(R)
    for key, sp in splits.items():
        Q = torch.as_tensor(sp["X"][:, cols])
        d = torch.cdist(Q, Rt).min(1).values.numpy()[:, None].astype(np.float32)
        sp["X"] = np.concatenate([sp["X"], d], 1)
        sp["names"] = names + ["latent_nn_dist"]
        sp["feature_groups"] = {**sp["feature_groups"],
                                "apriori": sp["feature_groups"]["apriori"] + [sp["X"].shape[1] - 1]}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot", choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=None, help="run a single repeat seed")
    ap.add_argument("--stage", default="all", choices=["data", "wm", "features", "all"])
    args = ap.parse_args()

    cfg: ExpConfig = CONFIGS[args.config]
    seeds = (args.seed,) if args.seed is not None else cfg.seeds
    device = get_device()
    root = ROOT / "results" / cfg.name
    root.mkdir(parents=True, exist_ok=True)
    save_provenance(root / "provenance.json", cfg)
    log(f"config={cfg.name} device={device} seeds={seeds}")

    data = build_data(cfg, ROOT / "experiments" / cfg.name)
    if args.stage == "data":
        return

    for seed in seeds:
        set_seed(seed)
        log(f"=== repeat seed {seed} ===")
        models = build_ensemble(cfg, data, seed, ROOT / "checkpoints" / cfg.name, device)
        if args.stage == "wm":
            continue

        splits = {}
        for key in ["A_gtrain", "A_val", "A_test", "B_test", "C_test", "D_test"]:
            splits[key] = extract_split(models, data[key], cfg, device, key)
        add_latent_nn_distance(splits)

        payload = {f"{k}__{f}": v[f] for k, v in splits.items()
                   for f in ["X", "y", "y_nonmono", "y_m0", "E", "groups", "starts",
                             "priv", "acts"]}
        payload["names"] = np.array(splits["A_test"]["names"], dtype=object)
        np.savez_compressed(root / f"features_seed{seed}.npz", **payload)
        json.dump({k: v for k, v in splits["A_test"]["feature_groups"].items()},
                  open(root / f"feature_groups_seed{seed}.json", "w"))
        log(f"saved features for seed {seed}")


if __name__ == "__main__":
    main()
