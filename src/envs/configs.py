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
    obs_noise: float = 0.0
    action_dropout: float = 0.0
    family: str = "epistemic"   # taxonomy under test in H8/H9: epistemic | aleatoric


ENV_CONFIGS: dict[str, EnvConfig] = {
    "A": EnvConfig("A", "MiniGrid-DoorKey-16x16-v0", 0.0, "training distribution"),
    "B": EnvConfig("B", "MiniGrid-FourRooms-v0", 0.0, "unseen layout topology"),
    "C": EnvConfig("C", "MiniGrid-DoorKey-16x16-v0", 0.15, "modified stochastic dynamics",
                   family="aleatoric"),
    "D": EnvConfig("D", "MiniGrid-MultiRoom-N6-v0", 0.0, "long-horizon corridor chains"),
    # H8 taxonomy test: E is a NEW epistemic shift, F a NEW aleatoric one with a different
    # mechanism from C (unreliable observation rather than unreliable action).
    "E": EnvConfig("E", "MiniGrid-SimpleCrossingS11N5-v0", 0.0, "unseen maze topology",
                   family="epistemic"),
    "F": EnvConfig("F", "MiniGrid-DoorKey-16x16-v0", 0.0, "observation noise",
                   obs_noise=0.15, family="aleatoric"),
}


class ObsNoiseWrapper(gym.ObservationWrapper):
    """Corrupt a fraction of cells in the egocentric view to a random object id.

    This is a SECOND aleatoric shift, with a different mechanism from `SlipWrapper`: the
    dynamics are untouched and the *observation* is unreliable, rather than the action. The
    epistemic/aleatoric taxonomy in RESULTS.md predicts this should behave like config C
    (slip) and unlike the geometric shifts, which is exactly what config F tests.
    """

    def __init__(self, env, p: float, seed: int = 0):
        super().__init__(env)
        self.p = float(p)
        self._rng = np.random.default_rng(seed)

    def observation(self, obs):
        img = obs["image"].copy()
        mask = self._rng.random(img.shape[:2]) < self.p
        if mask.any():
            # object channel only; keep colour/state so the view stays well-formed
            img[..., 0][mask] = self._rng.integers(1, 10, size=int(mask.sum()))
        obs = dict(obs)
        obs["image"] = img
        return obs


class ActionDropoutWrapper(gym.Wrapper):
    """With probability `p` the requested action silently fails to execute.

    Uses MiniGrid's `done` action (index 6), verified to be a true no-op that does not
    terminate (`minigrid_env.py`: `elif action == self.actions.done: pass`). This is a
    THIRD aleatoric mechanism, distinct from slip (which substitutes a *different* action)
    and from observation noise (which corrupts perception, leaving dynamics intact).
    """

    DONE_ACTION = 6

    def __init__(self, env, p: float, seed: int = 0):
        super().__init__(env)
        self.p = float(p)
        self._rng = np.random.default_rng(seed)
        self.last_dropped = False

    def step(self, action):
        self.last_dropped = self.p > 0.0 and self._rng.random() < self.p
        if self.last_dropped:
            action = self.DONE_ACTION
        return self.env.step(action)


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
    if getattr(cfg, "action_dropout", 0.0) > 0.0:
        env = ActionDropoutWrapper(env, cfg.action_dropout, seed=slip_seed + 4242)
    if getattr(cfg, "obs_noise", 0.0) > 0.0:
        env = ObsNoiseWrapper(env, cfg.obs_noise, seed=slip_seed + 7777)
    return env


def aleatoric_config(name: str, kind: str, p: float) -> EnvConfig:
    """Ad-hoc config for the H9 severity ladder. `kind` is 'obs' or 'dropout'."""
    base = ENV_CONFIGS["A"]
    if kind == "obs":
        return EnvConfig(name, base.env_id, 0.0, f"observation noise p={p}",
                         obs_noise=p, family="aleatoric")
    if kind == "dropout":
        return EnvConfig(name, base.env_id, 0.0, f"action dropout p={p}",
                         action_dropout=p, family="aleatoric")
    raise ValueError(kind)
