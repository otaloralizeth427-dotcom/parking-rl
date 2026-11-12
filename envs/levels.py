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


def _with_defaults(kwargs: dict) -> dict:
    """Completa con los valores por defecto de ParkingEnv: una opción agregada después (p. ej.
    crossing_pedestrians) que falta en un manifiesto viejo vale su valor por defecto, que reproduce el
    comportamiento con que se entrenó ese modelo."""
    import inspect
    defaults = {k: v.default for k, v in inspect.signature(ParkingEnv.__init__).parameters.items()
                if v.default is not inspect.Parameter.empty and k != "config"}
    return {**defaults, **(kwargs or {})}


def model_matches_level(manifest: dict, level_id: str) -> bool:
    """Un modelo solo se sirve en la definición del nivel con la que se entrenó (mismas observaciones)."""
    level = load_level(level_id)
    return (manifest.get("environment_version") == level["environment_version"]
            and _with_defaults(manifest.get("environment_kwargs")) == _with_defaults(env_kwargs(level)))


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
            **({"pedestrians_evade": True} if level.get("pedestrians_evade") else {}),
            **({"layout": level["layout"]} if level.get("layout", "aisle") != "aisle" else {}),
            **({"route_start": level["route_start"]} if "route_start" in level else {})}
