"""Niveles de dificultad versionados (configs/levels/*.json)."""

from __future__ import annotations

import json
from pathlib import Path

from envs.parking_env import SUPPORTED_VERSIONS, ParkingEnv

LEVELS_DIR = Path(__file__).resolve().parents[1] / "configs" / "levels"
# Los niveles disponibles son los archivos presentes en configs/levels/
LEVEL_IDS = tuple(sorted(p.stem for p in LEVELS_DIR.glob("N*.json")))


def load_level(level_id: str) -> dict:
    if level_id not in LEVEL_IDS:
        raise ValueError(f"Nivel desconocido: {level_id!r}. Opciones: {LEVEL_IDS}")
    level = json.loads((LEVELS_DIR / f"{level_id}.json").read_text())
    if level["environment_version"] not in SUPPORTED_VERSIONS:
        raise ValueError(f"{level_id} pide el ambiente {level['environment_version']}, "
                         f"pero el código actual solo reproduce {SUPPORTED_VERSIONS}")
    return level


def model_matches_level(manifest: dict, level_id: str) -> bool:
    """Un modelo solo se sirve en la definición del nivel con la que se entrenó (mismas observaciones)."""
    level = load_level(level_id)
    return (manifest.get("environment_version") == level["environment_version"]
            and manifest.get("environment_kwargs") == env_kwargs(level))


def make_env(level_id: str) -> ParkingEnv:
    level = load_level(level_id)
    return ParkingEnv(**env_kwargs(level))


def env_kwargs(level: dict) -> dict:
    """Argumentos de ParkingEnv que define un nivel."""
    return {"difficulty": level["difficulty"], "sensor_noise": level["sensor_noise"],
            "pedestrians": level.get("pedestrians", 0), "cones": level.get("cones"),
            "blocking_cones": level.get("blocking_cones", 0), "bumper_sensors": level.get("bumper_sensors", 0),
            "crossing_pedestrians": level.get("crossing_pedestrians", 0),
            **({"pedestrians_yield": True} if level.get("pedestrians_yield") else {}),
            **({"pedestrians_evade": True} if level.get("pedestrians_evade") else {})}
