"""Environment configurations A-D and the slip (stochastic dynamics) wrapper.

All configs share one observation space: an egocentric symbolic view of size
AGENT_VIEW x AGENT_VIEW x 3. Because the view is egocentric, grid size is not part of
the observation and the world model transfers across configs by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

AGENT_VIEW = 5
MAX_STEPS = 400

# MiniGrid action ids we use. 'done' and 'drop' are excluded: 'done' is a no-op that
# terminates, 'drop' is only meaningful while carrying and adds no dynamics of interest.
ACTIONS = [0, 1, 2, 3, 5]  # left, right, forward, pickup, toggle
N_ACTIONS = len(ACTIONS)
A_LEFT, A_RIGHT, A_FORWARD, A_PICKUP, A_TOGGLE = range(N_ACTIONS)


@dataclass(frozen=True)
class EnvConfig:
    name: str
    env_id: str
    slip: float
    description: str


ENV_CONFIGS: dict[str, EnvConfig] = {
    "A": EnvConfig("A", "MiniGrid-DoorKey-16x16-v0", 0.0, "training distribution"),
    "B": EnvConfig("B", "MiniGrid-FourRooms-v0", 0.0, "unseen layout topology"),
    "C": EnvConfig("C", "MiniGrid-DoorKey-16x16-v0", 0.15, "modified stochastic dynamics"),
    "D": EnvConfig("D", "MiniGrid-MultiRoom-N6-v0", 0.0, "long-horizon corridor chains"),
}


class SlipWrapper(gym.Wrapper):
    """With probability `slip`, replace the requested action by a uniform random one.

    The replacement is drawn from a wrapper-owned RNG so that slip noise is reproducible
    and independent of the layout seed.
    """

    def __init__(self, env, slip: float, seed: int = 0):
        super().__init__(env)
        self.slip = float(slip)
        self._rng = np.random.default_rng(seed)
        self.last_slipped = False

    def step(self, action):
        self.last_slipped = False
        if self.slip > 0.0 and self._rng.random() < self.slip:
            action = int(self._rng.choice(ACTIONS))
            self.last_slipped = True
        return self.env.step(action)


def make_env(cfg: EnvConfig, slip_seed: int = 0):
    env = gym.make(cfg.env_id, agent_view_size=AGENT_VIEW, max_steps=MAX_STEPS)
    if cfg.slip > 0.0:
        env = SlipWrapper(env, cfg.slip, seed=slip_seed)
    return env
