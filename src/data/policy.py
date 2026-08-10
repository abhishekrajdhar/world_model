"""Scripted exploration policy for trajectory collection.

A purely random agent almost never picks up the key or opens the door in DoorKey, so the
world model would never see those dynamics. We therefore use a BFS-based scripted explorer
with epsilon-greedy noise: it pursues key -> door -> random reachable cells, taking a random
action with probability `epsilon`.

Planning is done in (x, y, dir) space with actions {left, right, forward}, which yields
exact shortest action sequences.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from minigrid.core.world_object import Door, Goal, Key, Lava

from src.envs.configs import A_FORWARD, A_LEFT, A_PICKUP, A_RIGHT, A_TOGGLE, ACTIONS

# MiniGrid direction vectors indexed by agent_dir: 0=right, 1=down, 2=left, 3=up
DIR_VEC = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int64)


def _passable(cell) -> bool:
    """Cells the planner is willing to walk onto.

    Goal and Lava are excluded because both terminate the episode, which would truncate
    trajectories and starve the dataset of long rollouts.
    """
    if cell is None:
        return True
    if isinstance(cell, (Goal, Lava)):
        return False
    return bool(cell.can_overlap())


def _grid_masks(env):
    u = env.unwrapped
    W, H = u.width, u.height
    passable = np.zeros((W, H), dtype=bool)
    for x in range(W):
        for y in range(H):
            passable[x, y] = _passable(u.grid.get(x, y))
    return passable


def bfs_actions(env, start_pos, start_dir, goal_test):
    """Shortest action sequence from (start_pos, start_dir) to any state satisfying
    `goal_test(pos, dir)`. Returns None if unreachable."""
    u = env.unwrapped
    W, H = u.width, u.height
    passable = _grid_masks(env)

    start = (int(start_pos[0]), int(start_pos[1]), int(start_dir))
    if goal_test((start[0], start[1]), start[2]):
        return []

    seen = {start}
    queue = deque([(start, [])])
    while queue:
        (x, y, d), path = queue.popleft()
        if len(path) > 4 * (W * H):
            continue
        for act in (A_LEFT, A_RIGHT, A_FORWARD):
            if act == A_LEFT:
                nxt = (x, y, (d - 1) % 4)
            elif act == A_RIGHT:
                nxt = (x, y, (d + 1) % 4)
            else:
                dx, dy = DIR_VEC[d]
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and 0 <= ny < H) or not passable[nx, ny]:
                    continue
                nxt = (nx, ny, d)
            if nxt in seen:
                continue
            seen.add(nxt)
            npath = path + [act]
            if goal_test((nxt[0], nxt[1]), nxt[2]):
                return npath
            queue.append((nxt, npath))
    return None


def _facing_test(target_cell):
    """Goal test: standing adjacent to `target_cell` and facing it."""
    tx, ty = target_cell

    def test(pos, d):
        dx, dy = DIR_VEC[d]
        return (pos[0] + dx, pos[1] + dy) == (tx, ty)

    return test


def _find_objects(env, cls):
    u = env.unwrapped
    out = []
    for x in range(u.width):
        for y in range(u.height):
            c = u.grid.get(x, y)
            if isinstance(c, cls):
                out.append(((x, y), c))
    return out


class ScriptedExplorer:
    """Forward-biased exploration with occasional scripted key/door subgoals.

    Design note (Phase 1 validation): an earlier version wandered via BFS to random reachable
    cells. That made `moved` almost deterministic -- P(moved | forward) = 0.985 -- because BFS
    paths never walk into walls, which removed the only source of epistemic uncertainty the
    experiment is designed to measure. The forward-biased random walk deliberately probes
    cells the agent has not verified are free, which is what makes the reliable-imagination
    horizon a non-trivial quantity. Scripted subgoals are retained (at probability
    `p_script`) so that key/door dynamics still appear in the data.
    """

    def __init__(self, epsilon: float = 0.3, seed: int = 0,
                 p_forward: float = 0.6, p_script: float = 0.35):
        self.epsilon = epsilon
        self.p_forward = p_forward
        self.p_script = p_script
        self.rng = np.random.default_rng(seed)
        self._plan: list[int] = []

    def reset(self):
        self._plan = []

    def _safe_random_action(self, env) -> int:
        """Forward-biased random action, never walking into Goal/Lava (both end the episode)."""
        u = env.unwrapped
        r = self.rng.random()
        if r < self.p_forward:
            act = A_FORWARD
        elif r < self.p_forward + (1 - self.p_forward) * 0.42:
            act = A_LEFT
        elif r < self.p_forward + (1 - self.p_forward) * 0.84:
            act = A_RIGHT
        else:
            act = int(self.rng.choice([A_PICKUP, A_TOGGLE]))
        if act == A_FORWARD:
            dx, dy = DIR_VEC[u.agent_dir]
            ahead = u.grid.get(int(u.agent_pos[0]) + dx, int(u.agent_pos[1]) + dy)
            if isinstance(ahead, (Goal, Lava)):
                act = int(self.rng.choice([A_LEFT, A_RIGHT]))
        return act

    def _make_plan(self, env) -> list[int]:
        u = env.unwrapped
        pos, d = u.agent_pos, u.agent_dir

        # 1. fetch a key if not carrying one
        if u.carrying is None:
            keys = _find_objects(env, Key)
            if keys:
                cell = keys[0][0]
                path = bfs_actions(env, pos, d, _facing_test(cell))
                if path is not None:
                    return path + [A_PICKUP]

        # 2. open a closed door
        doors = [(c, o) for c, o in _find_objects(env, Door) if not o.is_open]
        if doors:
            cell = doors[0][0]
            path = bfs_actions(env, pos, d, _facing_test(cell))
            if path is not None:
                return path + [A_TOGGLE]

        # 3. no scripted subgoal available -> explore
        return []

    def act(self, env) -> int:
        if self._plan:
            return self._plan.pop(0)
        if self.rng.random() < self.p_script:
            self._plan = self._make_plan(env)
            if self._plan:
                return self._plan.pop(0)
        return self._safe_random_action(env)
