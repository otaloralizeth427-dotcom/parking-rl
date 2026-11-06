"""Agente Q-learning tabular con discretización del estado."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.agent_base import MODELS_DIR, Agent

SENSOR_RANGE = 8.0
DANGER_DISTANCE = 1.0
FRONT_SENSORS = [7, 0, 1]
REAR_SENSORS = [3, 4, 5]


class StateDiscretizer:
    """Convierte la observación continua en un índice de la tabla Q."""

    def __init__(self):
        pos_edges = np.array([-6, -3, -1.5, -0.5, 0.5, 1.5, 3, 6], dtype=float)
        self.edges = {
            "dx": pos_edges,
            "dy": pos_edges,
            "heading": np.array([-1.2, -0.6, -0.25, 0.25, 0.6, 1.2]),
            "speed": np.array([-0.02, 0.02, 0.35]),
            "steer": np.array([-0.6, -0.1, 0.1, 0.6]),
        }
        self.dims = [len(e) + 1 for e in self.edges.values()] + [2, 2]  # + peligro adelante/atrás
        self.n_states = int(np.prod(self.dims))

    def __call__(self, obs: np.ndarray) -> int:
        sensors = obs[6:] * SENSOR_RANGE
        values = [obs[0] * 10, obs[1] * 10, obs[2] * np.pi / 2, obs[3], obs[4]]
        idx = [int(np.searchsorted(e, v)) for e, v in zip(self.edges.values(), values)]
        idx.append(int(sensors[FRONT_SENSORS].min() < DANGER_DISTANCE))
        idx.append(int(sensors[REAR_SENSORS].min() < DANGER_DISTANCE))
        return int(np.ravel_multi_index(idx, self.dims))


class QLearningAgent(Agent):
    name = "q_learning"
    default_path = MODELS_DIR / "q_table.npz"

    def __init__(self, n_actions: int = 5, alpha: float = 0.1, gamma: float = 0.98, **kwargs):
        super().__init__(n_actions, **kwargs)
        if not 0 < alpha <= 1 or not 0 < gamma <= 1:
            raise ValueError("alpha y gamma deben estar en (0, 1]")
        self.alpha = alpha
        self.gamma = gamma
        self.discretize = StateDiscretizer()
        self.q_table = np.zeros((self.discretize.n_states, n_actions), dtype=np.float32)

    def q_values(self, obs):
        return self.q_table[self.discretize(obs)]

    def learn(self, obs, action, reward, next_obs, terminated):
        s, s_next = self.discretize(obs), self.discretize(next_obs)
        target = reward if terminated else reward + self.gamma * self.q_table[s_next].max()
        self.q_table[s, action] += self.alpha * (target - self.q_table[s, action])

    def visited_fraction(self) -> float:
        return float(np.any(self.q_table != 0, axis=1).mean())

    def save(self, path=None):
        path = Path(path or self.default_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q_table=self.q_table, alpha=self.alpha, gamma=self.gamma,
                            epsilon=self.epsilon, episodes=self.episodes_trained)
        return path

    @classmethod
    def load(cls, path=None):
        path = Path(path or cls.default_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe la tabla Q en {path}. Entrena primero el agente.")
        data = np.load(path)
        agent = cls(int(data["q_table"].shape[1]), alpha=float(data["alpha"]), gamma=float(data["gamma"]))
        if data["q_table"].shape != agent.q_table.shape:
            raise ValueError("La tabla Q guardada no coincide con la discretización actual")
        agent.q_table = data["q_table"]
        agent.epsilon = float(data["epsilon"])
        agent.episodes_trained = int(data["episodes"])
        return agent
