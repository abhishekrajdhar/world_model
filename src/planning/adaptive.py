"""Phase 11: adaptive-horizon planning.

Random-shooting MPC on top of the same world model. The only thing that varies between arms
is the imagination horizon H:

    fixed_H     : always imagine H steps
    adaptive    : imagine round(clip(g_phi(s_t), 1, H_MAX)) steps

Compute is measured as imagined states per decision (= n_samples * H), so an adaptive arm
that shortens its horizon where the model is unreliable spends strictly less compute.

Task: reach a target cell. The world model only predicts `dir`/`moved`, so a candidate
action sequence is scored by the closest approach of its imagined displacement to the target
displacement -- no reward model is required, which keeps the planner honest about what the
world model actually knows.
"""
from __future__ import annotations

import numpy as np
import torch

from src.data.dataset import gather
from src.envs.configs import ACTIONS, A_FORWARD, N_ACTIONS, make_env
from src.horizon.rih import CONTEXT, H_MAX, rollout
from src.models.world_model import integrate_displacement
from src.uncertainty.signals import extract_features


def _bfs_distances(passable: np.ndarray, start: tuple[int, int]) -> dict:
    """4-connected shortest-path distance from `start` over passable cells."""
    from collections import deque
    W, H = passable.shape
    dist = {start: 0}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < W and 0 <= ny < H and passable[nx, ny] and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    dist.pop(start, None)
    return dist


class RollingWindow:
    """Keeps the last CONTEXT observations/actions so the exact training-time feature
    extractor can be reused verbatim during planning."""

    def __init__(self, context: int = CONTEXT, h_max: int = H_MAX):
        self.context, self.h_max = context, h_max
        self.buf: list[dict] = []

    def push(self, image, direction, carrying, door_open, moved, action, pos):
        self.buf.append({"image": image, "dir": direction, "carrying": carrying,
                         "door_open": door_open, "moved": moved, "action": action, "pos": pos})
        if len(self.buf) > self.context:
            self.buf.pop(0)

    def ready(self) -> bool:
        return len(self.buf) == self.context

    def batch(self, device, pad_to: int) -> dict:
        """Window of length `pad_to` (>= context); future slots are padded and never read."""
        out = {}
        pad = pad_to - self.context
        for k in ["image", "dir", "carrying", "door_open", "moved", "action", "pos"]:
            vals = [b[k] for b in self.buf]
            arr = np.stack(vals + [vals[-1]] * pad)
            out[k] = torch.as_tensor(arr[None].astype(np.int64), device=device)
        return out


def sample_actions(rng, n, H, p_forward=0.6):
    """Sample candidate sequences from the SAME forward-biased distribution used to collect
    training data. Uniform sampling would push the world model (and g_phi, which was fitted
    under the collection distribution) off-distribution, confounding the comparison."""
    r = rng.random((n, H))
    a = np.full((n, H), A_FORWARD, dtype=np.int64)
    rest = (1 - p_forward)
    a[r >= p_forward] = 0
    turn = (r >= p_forward) & (r < p_forward + rest * 0.42)
    right = (r >= p_forward + rest * 0.42) & (r < p_forward + rest * 0.84)
    other = r >= p_forward + rest * 0.84
    a[turn] = 0
    a[right] = 1
    a[other] = rng.integers(3, N_ACTIONS, size=int(other.sum()))
    a[r < p_forward] = A_FORWARD
    return a


@torch.no_grad()
def _best_action(models, win_batch, target_disp, horizon, n_samples, rng, device):
    """Random-shooting: sample action sequences, imagine, score by closest approach."""
    H = int(horizon)
    acts = torch.as_tensor(sample_actions(rng, n_samples, H), dtype=torch.long, device=device)
    big = {k: v.expand(n_samples, *v.shape[1:]).contiguous() for k, v in win_batch.items()}
    r = rollout(models, big, CONTEXT, H, actions=acts, need_truth=False)
    disp = r["disp_ens"].float()                                   # (N,H,2)
    tgt = torch.as_tensor(target_disp, dtype=torch.float32, device=device)
    dist = (disp - tgt).abs().sum(-1)                              # (N,H)
    score = dist.min(1).values
    best = int(score.argmin().item())
    return int(acts[best, 0].item()), n_samples * H


@torch.no_grad()
def run_episode(models, env_cfg, layout_seed, horizon_spec, n_samples=64, budget=80,
                device="cpu", predictor=None, standardizer=None, cols=None, rng_seed=0,
                warmup=120):
    """One planning episode. `horizon_spec` is an int (fixed) or 'adaptive'."""
    env = make_env(env_cfg, slip_seed=rng_seed)
    obs, _ = env.reset(seed=int(layout_seed))
    u = env.unwrapped
    rng = np.random.default_rng(rng_seed)

    from minigrid.core.world_object import Door

    def doors_open():
        for x in range(u.width):
            for y in range(u.height):
                c = u.grid.get(x, y)
                if isinstance(c, Door) and c.is_open:
                    return 1
        return 0

    win = RollingWindow()

    # WARM-UP. In DoorKey the door starts locked, so BFS from the start reaches only one
    # convex room, where shortest-path distance equals Manhattan distance and no detour
    # target exists (this produced 0/60 usable layouts in the first run). Run the scripted
    # explorer first so the key is collected and the door opened; then both rooms are
    # reachable and crossing the doorway is a genuine detour. It also warms the belief state.
    from src.data.policy import ScriptedExplorer, _grid_masks
    warm = ScriptedExplorer(seed=rng_seed, p_forward=0.6, p_script=0.9)
    prev = np.array(u.agent_pos, dtype=np.int64)
    moved = 0
    for _ in range(warmup):
        win.push(obs["image"].astype(np.int64), int(u.agent_dir), int(u.carrying is not None),
                 doors_open(), moved, 0, np.array(u.agent_pos, dtype=np.int64))
        obs, _, term, trunc, _ = env.step(ACTIONS[warm.act(env)])
        cur = np.array(u.agent_pos, dtype=np.int64)
        moved = int((cur != prev).any())
        prev = cur
        if term or trunc:
            env.close()
            return None
        if doors_open() and win.ready():
            break

    # Pick a target that REQUIRES A DETOUR: shortest-path distance must exceed the Manhattan
    # distance, so greedy one-step lookahead cannot solve it and planning depth matters.
    # Tiers relax the requirement; the tier actually used is reported, never silently.
    passable = _grid_masks(env)
    start = np.array(u.agent_pos, dtype=np.int64)
    dist = _bfs_distances(passable, tuple(start))
    tier_used, cand = None, []
    for tier, (min_detour, lo, hi) in enumerate([(6, 8, 26), (4, 8, 24), (2, 6, 20), (0, 6, 18)]):
        cand = [(c, d) for c, d in dist.items()
                if lo <= d <= hi
                and d - (abs(c[0] - start[0]) + abs(c[1] - start[1])) >= min_detour]
        if cand:
            tier_used = tier
            break
    if not cand:
        env.close()
        return None
    choice = cand[rng.integers(len(cand))]
    target = np.array(choice[0], dtype=np.int64)
    detour = int(choice[1] - (abs(target[0] - start[0]) + abs(target[1] - start[1])))

    prev_pos = start.copy()
    imagined_total, decisions, horizons_used = 0, 0, []
    reached, steps = False, 0

    for t in range(budget):
        cur = np.array(u.agent_pos, dtype=np.int64)
        if (cur == target).all():
            reached = True
            break
        win.push(obs["image"].astype(np.int64), int(u.agent_dir), int(u.carrying is not None),
                 doors_open(), moved, 0, cur)
        if not win.ready():
            a_idx = int(rng.integers(N_ACTIONS))
        else:
            wb = win.batch(device, pad_to=CONTEXT + H_MAX + 1)
            if horizon_spec == "adaptive":
                r = rollout(models, wb, CONTEXT, H_MAX, need_truth=False)
                # Charge adaptive for estimating its own horizon: this probe rollout costs
                # K * H_MAX imagined states. Ignoring it would flatter the adaptive arm.
                imagined_total += len(models) * H_MAX
                X, names, _ = extract_features(models, wb, r, CONTEXT, H_MAX)
                # the planner needs the same extra column the analysis appended
                if cols is not None and X.shape[1] < max(cols) + 1:
                    X = np.concatenate([X, np.zeros((len(X), max(cols) + 1 - X.shape[1]),
                                                    np.float32)], 1)
                xt = torch.as_tensor(standardizer(X[:, cols]), device=device)
                h_use = int(np.clip(round(float(predictor.predict(xt)[0][0])), 1, H_MAX))
            else:
                h_use = int(horizon_spec)
            horizons_used.append(h_use)
            a_idx, cost = _best_action(models, wb, target - cur, h_use, n_samples, rng, device)
            imagined_total += cost
            decisions += 1

        obs, _, term, trunc, _ = env.step(ACTIONS[a_idx])
        new_pos = np.array(u.agent_pos, dtype=np.int64)
        moved = int((new_pos != prev_pos).any())
        prev_pos = new_pos
        steps = t + 1
        if term or trunc:
            break
    if (np.array(u.agent_pos, dtype=np.int64) == target).all():
        reached = True
    env.close()
    return {
        "reached": bool(reached), "steps": steps,
        "imagined_states": imagined_total,
        "imagined_per_decision": imagined_total / max(1, decisions),
        "mean_horizon": float(np.mean(horizons_used)) if horizons_used else 0.0,
        "final_dist": int(np.abs(np.array(u.agent_pos) - target).sum()),
        "start_dist": int(np.abs(start - target).sum()),
        "path_dist": int(choice[1]),
        "detour": detour,
        "tier": tier_used,
    }
