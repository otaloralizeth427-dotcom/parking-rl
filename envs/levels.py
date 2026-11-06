"""Niveles de dificultad versionados (configs/levels/*.json)."""

from __future__ import annotations

import json
from pathlib import Path

from envs.parking_env import ENV_VERSION, ParkingEnv

LEVELS_DIR = Path(__file__).resolve().parents[1] / "configs" / "levels"
# Los niveles disponibles son los archivos presentes en configs/levels/
LEVEL_IDS = tuple(sorted(p.stem for p in LEVELS_DIR.glob("N*.json")))


def load_level(level_id: str) -> dict:
    if level_id not in LEVEL_IDS:
        raise ValueError(f"Nivel desconocido: {level_id!r}. Opciones: {LEVEL_IDS}")
    level = json.loads((LEVELS_DIR / f"{level_id}.json").read_text())
    if level["environment_version"] != ENV_VERSION:
        raise ValueError(f"{level_id} pide el ambiente {level['environment_version']}, "
                         f"pero el código actual es {ENV_VERSION}")
    return level


def make_env(level_id: str) -> ParkingEnv:
    level = load_level(level_id)
    return ParkingEnv(**env_kwargs(level))


def env_kwargs(level: dict) -> dict:
    """Argumentos de ParkingEnv que define un nivel."""
    return {"difficulty": level["difficulty"], "sensor_noise": level["sensor_noise"],
            "pedestrians": level.get("pedestrians", 0), "cones": level.get("cones")}
