"""Phase 2: trajectory collection.

Trajectories are stored flat (all timesteps concatenated) with per-episode offsets, which
keeps loading fast and makes episode-clustered bootstrapping straightforward.

Recorded per timestep t:
    image      egocentric symbolic observation (5,5,3)
    dir        absolute agent direction in {0,1,2,3}
    pos        absolute (x,y) -- used only to build ground-truth targets, never fed to the model
    carrying   whether the agent holds an object
    door_open  whether any door in the grid is open
    moved      whether the transition INTO t changed the agent's cell (moved[0] = False)
    action     the action REQUESTED at t (before slip). This is what the agent knows and
               what the world model is conditioned on; slip is genuine aleatoric noise.
    slipped    whether the slip wrapper overrode the action at t
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from minigrid.core.world_object import Door

from src.data.policy import ScriptedExplorer
from src.envs.configs import ACTIONS, EnvConfig, make_env


PRIV_R = 5  # privileged patch radius -> (2*PRIV_R+1)^2 cells


@dataclass
class Trajectories:
    image: np.ndarray  # (N,5,5,3) uint8
    dir: np.ndarray  # (N,) int8
    pos: np.ndarray  # (N,2) int16
    priv: np.ndarray  # (N,11,11) uint8 -- TRUE local grid, privileged upper-bound only
    carrying: np.ndarray  # (N,) bool
    door_open: np.ndarray  # (N,) bool
    moved: np.ndarray  # (N,) bool
    action: np.ndarray  # (N,) int8
    slipped: np.ndarray  # (N,) bool
    ep_start: np.ndarray  # (E,) int64
    ep_len: np.ndarray  # (E,) int64
    ep_seed: np.ndarray  # (E,) int64

    def __len__(self) -> int:
        return len(self.ep_len)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> "Trajectories":
        with np.load(path) as d:
            return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def _priv_patch(env, r: int = PRIV_R) -> np.ndarray:
    """TRUE object indices in a (2r+1)^2 window centred on the agent, absolute orientation.

    This is PRIVILEGED information (it reveals cells the agent cannot see). It is used only
    to fit the privileged upper bound in Phase 8, never as an input to g_phi.
    """
    from minigrid.core.constants import OBJECT_TO_IDX
    u = env.unwrapped
    ax, ay = int(u.agent_pos[0]), int(u.agent_pos[1])
    out = np.full((2 * r + 1, 2 * r + 1), OBJECT_TO_IDX["wall"], dtype=np.uint8)
    for i, x in enumerate(range(ax - r, ax + r + 1)):
        for j, y in enumerate(range(ay - r, ay + r + 1)):
            if 0 <= x < u.width and 0 <= y < u.height:
                c = u.grid.get(x, y)
                out[i, j] = OBJECT_TO_IDX["empty"] if c is None else OBJECT_TO_IDX[c.type]
    return out


def _doors_open(env) -> bool:
    u = env.unwrapped
    for x in range(u.width):
        for y in range(u.height):
            c = u.grid.get(x, y)
            if isinstance(c, Door) and c.is_open:
                return True
    return False


def collect(
    cfg: EnvConfig,
    layout_seeds: list[int] | np.ndarray,
    steps_per_episode: int = 120,
    epsilon: float = 0.3,
    policy_seed: int = 0,
    min_length: int = 40,
    verbose: bool = False,
) -> Trajectories:
    env = make_env(cfg, slip_seed=policy_seed)
    policy = ScriptedExplorer(epsilon=epsilon, seed=policy_seed)

    bufs: dict[str, list] = {k: [] for k in
                            ["image", "dir", "pos", "priv", "carrying", "door_open", "moved",
                             "action", "slipped"]}
    ep_start, ep_len, ep_seed = [], [], []
    cursor = 0

    for i, seed in enumerate(layout_seeds):
        obs, _ = env.reset(seed=int(seed))
        policy.reset()
        u = env.unwrapped

        ep = {k: [] for k in bufs}
        prev_pos = np.array(u.agent_pos, dtype=np.int16)
        moved = False

        for t in range(steps_per_episode):
            ep["image"].append(obs["image"].astype(np.uint8))
            ep["dir"].append(np.int8(u.agent_dir))
            ep["pos"].append(np.array(u.agent_pos, dtype=np.int16))
            ep["priv"].append(_priv_patch(env))
            ep["carrying"].append(u.carrying is not None)
            ep["door_open"].append(_doors_open(env))
            ep["moved"].append(moved)

            a_idx = policy.act(env)
            ep["action"].append(np.int8(a_idx))

            obs, _, terminated, truncated, _ = env.step(ACTIONS[a_idx])
            ep["slipped"].append(bool(getattr(env, "last_slipped", False)))

            cur_pos = np.array(u.agent_pos, dtype=np.int16)
            moved = bool((cur_pos != prev_pos).any())
            prev_pos = cur_pos
            if terminated or truncated:
                break

        T = len(ep["image"])
        if T < min_length:
            continue
        for k in bufs:
            bufs[k].append(np.asarray(ep[k]))
        ep_start.append(cursor)
        ep_len.append(T)
        ep_seed.append(int(seed))
        cursor += T
        if verbose and (i + 1) % 250 == 0:
            print(f"  collected {i+1}/{len(layout_seeds)} episodes, {cursor} steps", flush=True)

    env.close()
    if not ep_len:
        raise RuntimeError(f"No episodes of length >= {min_length} collected for {cfg.name}")

    return Trajectories(
        image=np.concatenate(bufs["image"]).astype(np.uint8),
        dir=np.concatenate(bufs["dir"]).astype(np.int8),
        pos=np.concatenate(bufs["pos"]).astype(np.int16),
        priv=np.concatenate(bufs["priv"]).astype(np.uint8),
        carrying=np.concatenate(bufs["carrying"]).astype(bool),
        door_open=np.concatenate(bufs["door_open"]).astype(bool),
        moved=np.concatenate(bufs["moved"]).astype(bool),
        action=np.concatenate(bufs["action"]).astype(np.int8),
        slipped=np.concatenate(bufs["slipped"]).astype(bool),
        ep_start=np.asarray(ep_start, dtype=np.int64),
        ep_len=np.asarray(ep_len, dtype=np.int64),
        ep_seed=np.asarray(ep_seed, dtype=np.int64),
    )
