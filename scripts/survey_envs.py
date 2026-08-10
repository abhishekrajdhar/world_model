"""Environment + policy survey (Phase 1 validation).

We need a regime where the agent's planned path leaves the region it can currently see,
because that is the only source of genuine epistemic uncertainty about `moved`. This script
measures, WITHOUT any neural network, whether a candidate (env, policy) pair produces that
regime: how often forward moves are blocked, and how far the agent travels in H_max steps
relative to how far it can currently see.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402
from minigrid.core.world_object import Goal, Lava  # noqa: E402

from src.data.policy import DIR_VEC  # noqa: E402

ACTIONS = [0, 1, 2, 3, 5]
A_LEFT, A_RIGHT, A_FORWARD = 0, 1, 2

CANDIDATES = [
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-MultiRoom-N4-S5-v0",
    "MiniGrid-MultiRoom-N6-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-SimpleCrossingS9N3-v0",
    "MiniGrid-SimpleCrossingS11N5-v0",
    "MiniGrid-DoorKey-16x16-v0",
]


def forward_biased_action(rng, p_forward):
    r = rng.random()
    if r < p_forward:
        return A_FORWARD
    if r < p_forward + (1 - p_forward) / 2:
        return A_LEFT
    return A_RIGHT


def run(env_id, view, p_forward, n_ep=25, steps=120, seed0=0, h_max=30):
    try:
        env = gym.make(env_id, agent_view_size=view, max_steps=2000)
        env.reset(seed=0)
    except Exception as e:
        return {"env": env_id, "error": f"{type(e).__name__}: {e}"[:70]}
    rng = np.random.default_rng(0)

    moved_f, disps, ep_lens, blocked_unseen = [], [], [], []

    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed0 + ep)
        u = env.unwrapped
        pos_hist = []
        for t in range(steps):
            pos_hist.append(tuple(int(v) for v in u.agent_pos))
            a = forward_biased_action(rng, p_forward)
            if a == A_FORWARD:
                dx, dy = DIR_VEC[u.agent_dir]
                ahead = u.grid.get(int(u.agent_pos[0]) + dx, int(u.agent_pos[1]) + dy)
                if isinstance(ahead, (Goal, Lava)):
                    a = int(rng.choice([A_LEFT, A_RIGHT]))
            prev = tuple(int(v) for v in u.agent_pos)
            obs, _, term, trunc, _ = env.step(ACTIONS[a])
            cur = tuple(int(v) for v in u.agent_pos)
            if a == A_FORWARD:
                moved_f.append(cur != prev)
            if term or trunc:
                break
        ep_lens.append(len(pos_hist))
        for t in range(0, max(0, len(pos_hist) - h_max), 5):
            p0 = np.array(pos_hist[t]); pH = np.array(pos_hist[t + h_max])
            disps.append(np.abs(pH - p0).sum())
    env.close()
    return {
        "env": env_id, "view": view, "p_fwd": p_forward,
        "P(moved|fwd)": float(np.mean(moved_f)) if moved_f else float("nan"),
        "mean|disp_30|": float(np.mean(disps)) if disps else float("nan"),
        "max|disp_30|": float(np.max(disps)) if disps else float("nan"),
        "mean_ep_len": float(np.mean(ep_lens)),
        "visible_ahead": view - 1,
    }


if __name__ == "__main__":
    rows = []
    for env_id in CANDIDATES:
        for view, pf in [(5, 0.30), (5, 0.60), (5, 0.80), (3, 0.60)]:
            rows.append(run(env_id, view, pf))
    hdr = (f"{'env':32s} {'view':>4s} {'p_fwd':>5s} {'P(moved|fwd)':>12s} "
           f"{'|disp30|':>9s} {'max':>4s} {'eplen':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['env']:32s}  ERROR {r['error']}")
            continue
        print(f"{r['env']:32s} {r['view']:4d} {r['p_fwd']:5.2f} {r['P(moved|fwd)']:12.3f} "
              f"{r['mean|disp_30|']:9.2f} {r['max|disp_30|']:4.0f} {r['mean_ep_len']:6.0f}")
