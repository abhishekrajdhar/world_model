"""H7: is the metacognition result a property of world models, or of one implementation?

Three architectures, all trained on config A with identical data, seeds and horizon targets:

  M1  gru_large    GRU-RSSM, deter 512 / stoch 48        (the main-run architecture)
  M2  gru_small    GRU-RSSM, deter 256 / stoch 16        (same family, different capacity)
  M3  transformer  causal attention, no recurrence,      (different family entirely)
                   no autoregressive latent feedback

Two questions:
  (a) WITHIN each architecture -- do H2 (beats the Claim-A ceiling) and the key H6 cell
      (A+B -> D restores transfer) replicate?
  (b) ACROSS architectures -- does g_phi fitted on one architecture's features predict
      ANOTHER architecture's reliable horizon? The `no_latent` feature set has identical
      dimensionality and semantics across architectures, so this is a direct test.
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

from configs.experiment import CONFIGS as EXP_CONFIGS  # noqa: E402
from src.data.collect import Trajectories, collect  # noqa: E402
from src.envs.configs import ENV_CONFIGS  # noqa: E402
from src.evaluation.analysis import VARIANTS, cols_for  # noqa: E402
from src.evaluation.baselines import isotonic_fit_predict  # noqa: E402
from src.evaluation.stats import full_metrics, mae, paired_diff_ci  # noqa: E402
from src.horizon.predictor import train_predictor  # noqa: E402
from src.horizon.transfer import build_pools  # noqa: E402
from src.models.train import train_member  # noqa: E402
from src.models.transformer_wm import TransformerWMConfig, TransformerWorldModel  # noqa: E402
from src.models.world_model import WMConfig, WorldModel, count_params  # noqa: E402
from src.uncertainty.signals import SCALAR_SIGNALS  # noqa: E402
from src.utils.seeding import get_device, set_seed  # noqa: E402
from scripts.run_experiment import add_latent_nn_distance, build_data, extract_split  # noqa: E402

SPLITS = ["A_gtrain", "A_val", "A_test", "B_test", "C_test", "D_test"]

ARCHS = {
    "gru_large": ("gru", dict(deter=512, stoch=48, hidden=512, obs_embed=384)),
    "gru_small": ("gru", dict(deter=256, stoch=16, hidden=256, obs_embed=192)),
    "transformer": ("tfm", dict(d_model=256, n_layers=2, n_heads=4, stoch=32,
                                hidden=256, obs_embed=256)),
}


def make_model(kind: str, kw: dict):
    return WorldModel(WMConfig(**kw)) if kind == "gru" else TransformerWorldModel(
        TransformerWMConfig(**kw))


def train_arch(arch: str, data, cfg, seed: int, k: int, steps: int, device, ckpt_dir: Path):
    kind, kw = ARCHS[arch]
    models = []
    for m_i in range(k):
        p = ckpt_dir / f"{arch}_seed{seed}_member{m_i}.pt"
        if p.exists():
            ck = torch.load(p, map_location=device, weights_only=False)
            model = make_model(kind, kw).to(device)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            models.append(model)
            continue
        set_seed(seed * 100 + m_i)
        proto = make_model(kind, kw).to(device)
        t0 = time.time()
        model, _ = train_member(
            data["A_wmtrain"], data["A_val"], seed=seed * 100 + m_i, steps=steps,
            batch_size=cfg.wm_batch, seq_len=cfg.wm_seq_len, lr=cfg.wm_lr, device=device,
            amp=(device.type == "cuda"), wm_cfg=None, ckpt_path=None, verbose=False,
            log_every=steps, model=proto)
        model.eval()
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "arch": arch}, p)
        print(f"  [{arch}] member {m_i}: {count_params(model):,} params "
              f"({time.time()-t0:.0f}s)", flush=True)
        models.append(model)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the interface")
    args = ap.parse_args()

    cfg = EXP_CONFIGS[args.config]
    device = get_device()
    out = ROOT / "results" / "cross_arch"
    out.mkdir(parents=True, exist_ok=True)
    ck = ROOT / "checkpoints" / "cross_arch"
    print(f"device={device} archs={list(ARCHS)} seeds={args.seeds} K={args.k}", flush=True)

    if args.smoke:   # validate the interface, not the science
        cfg.max_anchors_per_split = 600
        cfg.anchor_stride = 32
    data = build_data(cfg, ROOT / "experiments" / cfg.name)

    per_arch: dict[str, dict] = {}
    for arch in ARCHS:
        per_arch[arch] = {}
        for seed in args.seeds:
            print(f"=== {arch} seed {seed} ===", flush=True)
            models = train_arch(arch, data, cfg, seed, args.k, args.steps, device, ck)
            splits = {k: extract_split(models, data[k], cfg, device, f"{arch}:{k}")
                      for k in SPLITS}
            add_latent_nn_distance(splits)
            per_arch[arch][seed] = splits
            if args.smoke:
                break
        if args.smoke:
            continue

    # Feature-vector layouts differ across architectures because the latent dimensionality
    # differs (deter 512/256/d_model 256), so column INDICES are not comparable. Select each
    # architecture's own `no_latent` columns and assert the resulting NAMES are identical --
    # that is what makes cross-architecture application of g_phi meaningful.
    arch_cols, sel_names = {}, None
    for arch in ARCHS:
        s = per_arch[arch][args.seeds[0]]["A_test"]
        c = cols_for(s["feature_groups"], VARIANTS["no_latent"], s["X"].shape[1])
        nm = [s["names"][i] for i in c]
        if sel_names is None:
            sel_names = nm
        elif nm != sel_names:
            raise SystemExit(f"feature-name mismatch for {arch}:\n"
                             f"  expected {sel_names[:8]}...\n  got      {nm[:8]}...")
        arch_cols[arch] = c
    print(f"\nshared feature subspace: {len(sel_names)} columns, names identical across "
          f"all {len(ARCHS)} architectures", flush=True)
    if args.smoke:
        print("\nSMOKE OK: all architectures produce the same feature interface")
        for arch in ARCHS:
            s = per_arch[arch][args.seeds[0]]["A_test"]
            print(f"  {arch:12s} n_feat={s['X'].shape[1]} H*={s['y'].mean():.2f} "
                  f"sd={s['y'].std():.2f}")
        return

    results: dict = {"archs": {}, "cross": {}, "h6_cell": {}}

    # ---------- (a) within-architecture replication ----------
    for arch in ARCHS:
        rec = []
        for seed in args.seeds:
            S = per_arch[arch][seed]
            pools = build_pools(S, arch_cols[arch])
            # H2: g_phi vs the optimistically-fitted Claim-A ceiling on A-test
            m, std, _ = train_predictor(
                pools["A"]["train"]["X"], pools["A"]["train"]["y"],
                pools["A"]["val"]["X"], pools["A"]["val"]["y"], seed=seed,
                epochs=cfg.g_epochs, hidden=cfg.g_hidden, depth=cfg.g_depth, lr=cfg.g_lr,
                device=str(device))
            te = pools["A"]["test"]
            p = m.predict(torch.as_tensor(std(te["X"]), device=device))[0]
            # Feature indices are architecture-specific (latent dims differ), so the name
            # list MUST come from this architecture, not from a shared one.
            nm = S["A_test"]["names"]
            best = None
            for lab, feat in SCALAR_SIGNALS.items():
                if feat not in nm:
                    continue
                u = S["A_test"]["X"][:, nm.index(feat)]
                c = isotonic_fit_predict(u, S["A_test"]["y"], u)   # fitted ON TEST: optimistic
                if best is None or mae(c, S["A_test"]["y"]) < mae(best[1], S["A_test"]["y"]):
                    best = (lab, c)
            met = full_metrics(p, te["y"], te["groups"], n_boot=cfg.n_boot, seed=seed)
            d = paired_diff_ci(mae, te["groups"], p, best[1], te["y"],
                               n_boot=cfg.n_boot, seed=seed)
            # H6 key cell: A vs A+B, evaluated on held-out D
            trAB = {k: np.concatenate([pools["A"]["train"][k], pools["B"]["train"][k]])
                    for k in ("X", "y", "groups")}
            vaAB = {k: np.concatenate([pools["A"]["val"][k], pools["B"]["val"][k]])
                    for k in ("X", "y", "groups")}
            m2, std2, _ = train_predictor(trAB["X"], trAB["y"], vaAB["X"], vaAB["y"], seed=seed,
                                          epochs=cfg.g_epochs, hidden=cfg.g_hidden,
                                          depth=cfg.g_depth, lr=cfg.g_lr, device=str(device))
            dte = pools["D"]["test"]
            pA = m.predict(torch.as_tensor(std(dte["X"]), device=device))[0]
            pAB = m2.predict(torch.as_tensor(std2(dte["X"]), device=device))[0]
            rec.append({
                "seed": seed,
                "A_test_mae": met["mae"]["point"], "A_test_rho": met["spearman"]["point"],
                "ceiling_signal": best[0],
                "vs_ceiling": d,
                "D_rho_trainA": full_metrics(pA, dte["y"], dte["groups"], n_boot=500,
                                             seed=seed)["spearman"]["point"],
                "D_rho_trainAB": full_metrics(pAB, dte["y"], dte["groups"], n_boot=500,
                                              seed=seed)["spearman"]["point"],
                "D_pred_trainA": float(pA.mean()), "D_pred_trainAB": float(pAB.mean()),
                "D_actual": float(dte["y"].mean()),
                "H_star_A_test": float(S["A_test"]["y"].mean()),
            })
        results["archs"][arch] = rec

    # ---------- (b) cross-architecture transfer of g_phi ----------
    for seed in args.seeds:
        pools = {a: build_pools(per_arch[a][seed], arch_cols[a]) for a in ARCHS}
        fitted = {}
        for a in ARCHS:
            m, std, _ = train_predictor(
                pools[a]["A"]["train"]["X"], pools[a]["A"]["train"]["y"],
                pools[a]["A"]["val"]["X"], pools[a]["A"]["val"]["y"], seed=seed,
                epochs=cfg.g_epochs, hidden=cfg.g_hidden, depth=cfg.g_depth, lr=cfg.g_lr,
                device=str(device))
            fitted[a] = (m, std)
        for src in ARCHS:
            for tgt in ARCHS:
                m, std = fitted[src]
                te = pools[tgt]["A"]["test"]
                p = m.predict(torch.as_tensor(std(te["X"]), device=device))[0]
                met = full_metrics(p, te["y"], te["groups"], n_boot=cfg.n_boot, seed=seed)
                results["cross"].setdefault(f"{src}->{tgt}", []).append(
                    {"seed": seed, "mae": met["mae"]["point"],
                     "rho": met["spearman"]["point"],
                     "pred_mean": float(p.mean()), "actual_mean": float(te["y"].mean())})

    json.dump(results, open(out / "cross_arch.json", "w"), indent=2, default=float)

    # ---------- report ----------
    print("\n" + "=" * 72 + "\n H7 cross-architecture\n" + "=" * 72)
    print("\n### Within architecture (mean over seeds)\n")
    print("| arch | H* on A | A MAE | A rho | beats Claim-A ceiling | D rho (train A) | "
          "D rho (train A+B) |")
    print("|---|---|---|---|---|---|---|")
    for arch, rec in results["archs"].items():
        g = lambda k: float(np.mean([r[k] for r in rec]))  # noqa: E731
        won = all(r["vs_ceiling"]["lo"] > 0 for r in rec)
        margin = float(np.mean([r["vs_ceiling"]["point"] for r in rec]))
        print(f"| {arch} | {g('H_star_A_test'):.2f} | {g('A_test_mae'):.3f} | "
              f"{g('A_test_rho'):+.3f} | {'yes' if won else 'NO'} (+{margin:.3f}) | "
              f"{g('D_rho_trainA'):+.3f} | {g('D_rho_trainAB'):+.3f} |")

    print("\n### Cross-architecture: g_phi fitted on SRC, evaluated on TGT's A-test\n")
    print("| fitted on \\ evaluated on | " + " | ".join(ARCHS) + " |")
    print("|---|" + "---|" * len(ARCHS))
    for src in ARCHS:
        cells = []
        for tgt in ARCHS:
            v = results["cross"][f"{src}->{tgt}"]
            rho = float(np.mean([x["rho"] for x in v]))
            mae_ = float(np.mean([x["mae"] for x in v]))
            s = f"rho {rho:+.3f}, MAE {mae_:.2f}"
            cells.append(s if src == tgt else f"**{s}**")
        print(f"| {src} | " + " | ".join(cells) + " |")
    print("\n(bold = transferred across architectures; diagonal = same architecture)")


if __name__ == "__main__":
    main()
