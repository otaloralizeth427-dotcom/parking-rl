"""Interfaz común de agentes y bucle de entrenamiento compartido."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np

MODELS_DIR = Path(__file__).parent / "models"


class Agent(ABC):
    name: str = "agent"

    def __init__(self, n_actions: int, epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995, seed: int | None = None):
        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError("Se requiere 0 <= epsilon_end <= epsilon_start <= 1")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay debe estar en (0, 1]")
        self.n_actions = n_actions
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.episodes_trained = 0

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Política epsilon-greedy; con greedy=True siempre explota."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.q_values(obs)
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def end_episode(self) -> None:
        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    @abstractmethod
    def q_values(self, obs: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def learn(self, obs, action, reward, next_obs, terminated: bool) -> None: ...

    @abstractmethod
    def save(self, path: str | Path | None = None) -> Path: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path | None = None) -> "Agent": ...


def run_episode(env, agent: Agent | None, train: bool = False, greedy: bool = True,
                seed: int | None = None) -> dict:
    """Ejecuta un episodio completo. agent=None equivale a la política aleatoria."""
    obs, info = env.reset(seed=seed)
    total = 0.0
    while True:
        if agent is None:
            action = int(env.action_space.sample())
        else:
            action = agent.act(obs, greedy=greedy and not train)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if train:
            agent.learn(obs, action, reward, next_obs, terminated)
        total += reward
        obs = next_obs
        if terminated or truncated:
            break
    if train:
        agent.end_episode()
    return {"return": total, "success": info["success"], "collision": info["collision"],
            "steps": info["steps"], "distance": info["distance"],
            "heading_error": abs(info["heading_error"])}


def train(agent: Agent, env, episodes: int, seed: int = 0,
          callback: Callable[[int, dict], bool | None] | None = None,
          log_every: int = 0) -> list[dict]:
    """Entrena por episodios. El callback recibe (episodio, resultado) y puede devolver True para detener."""
    if episodes <= 0:
        raise ValueError("episodes debe ser positivo")
    env.action_space.seed(seed)
    history = []
    for ep in range(episodes):
        result = run_episode(env, agent, train=True, seed=seed + ep)
        result["epsilon"] = agent.epsilon
        history.append(result)
        if log_every and (ep + 1) % log_every == 0:
            recent = history[-log_every:]
            print(f"[{agent.name}] ep {ep + 1:5d}  "
                  f"éxito {np.mean([r['success'] for r in recent]):5.1%}  "
                  f"choques {np.mean([r['collision'] for r in recent]):5.1%}  "
                  f"retorno {np.mean([r['return'] for r in recent]):7.1f}  "
                  f"eps {agent.epsilon:.3f}", flush=True)
        if callback and callback(ep, result):
            break
    return history
