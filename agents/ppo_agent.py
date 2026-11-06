"""Agente PPO (Stable-Baselines3 + PyTorch) con la misma interfaz act() que los demás."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PPOAgent:
    name = "ppo"

    def __init__(self, model):
        self.model = model

    @classmethod
    def create(cls, env, seed: int, **hyperparams) -> "PPOAgent":
        from stable_baselines3 import PPO

        policy_kwargs = {"net_arch": hyperparams.pop("net_arch", [64, 64])}
        return cls(PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu",
                       policy_kwargs=policy_kwargs, **hyperparams))

    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        action, _ = self.model.predict(obs, deterministic=greedy)
        return int(action)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)          # Stable-Baselines3 agrega la extensión .zip
        return path.with_suffix(".zip")

    @classmethod
    def load(cls, path: str | Path) -> "PPOAgent":
        from stable_baselines3 import PPO

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el modelo PPO en {path}")
        return cls(PPO.load(path, device="cpu"))
