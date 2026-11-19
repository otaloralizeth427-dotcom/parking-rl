"""Núcleo de la evaluación con suites fijas, sin dependencias de gráficos.

Lo usan el pipeline (pipelines/evaluate.py) y el servidor publicado (evaluar y comparar desde la web),
así ambos miden exactamente lo mismo.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from envs.levels import make_env

ROOT = Path(__file__).resolve().parents[1]
SUITES_DIR = ROOT / "configs" / "suites"
OUTCOMES = ("success", "obstacle", "pedestrian", "boundary", "timeout")
OUTCOME_LABELS = {"success": "Éxito", "obstacle": "Choque con obstáculo", "pedestrian": "Atropello",
                  "boundary": "Salida del área", "timeout": "Tiempo agotado"}


def load_suite(suite_id: str, suites_dir: Path | None = None) -> dict:
    path = (suites_dir or SUITES_DIR) / f"{suite_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe la suite {suite_id} ({path})")
    suite = json.loads(path.read_text())
    if suite["episodes"] <= 0:
        raise ValueError("La suite debe tener al menos un episodio")
    return suite


def suite_for_level(level_id: str) -> dict:
    return load_suite(f"{level_id}-suite-v1")


def run_case(env, policy, seed: int) -> dict:
    """Un episodio con un modelo congelado; policy=None es la política aleatoria."""
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    total, latencies = 0.0, []
    while True:
        start = time.perf_counter()
        action = int(env.action_space.sample()) if policy is None else policy.act(obs, greedy=True)
        latencies.append((time.perf_counter() - start) * 1000)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    outcome = "success" if info["success"] else info["collision_kind"] or "timeout"
    return {
        "seed": seed, "outcome": outcome, "success": info["success"],
        "terminated": terminated, "truncated": truncated,
        "steps": info["steps"], "time_s": round(info["steps"] * env.cfg.dt, 2),
        "return": round(total, 3), "final_distance_m": round(info["distance"], 3),
        "final_heading_error_deg": round(float(np.degrees(abs(info["heading_error"]))), 2),
        "latency_ms": round(float(np.mean(latencies)), 4),
    }


def evaluate(policy, suite: dict) -> dict:
    env = make_env(suite["level_id"])
    cases = [run_case(env, policy, suite["seed_start"] + i) for i in range(suite["episodes"])]
    return {"suite_id": suite["suite_id"], "level_id": suite["level_id"],
            "summary": summarize(cases), "cases": cases}


def summarize(cases: list[dict]) -> dict:
    counts = Counter(c["outcome"] for c in cases)
    n = len(cases)
    mean = lambda key: round(float(np.mean([c[key] for c in cases])), 3)  # noqa: E731
    return {
        "episodes": n,
        "success_rate": round(counts["success"] / n, 4),
        "outcomes": {o: counts.get(o, 0) for o in OUTCOMES},
        "mean_return": mean("return"),
        "mean_time_s": mean("time_s"),
        "mean_final_distance_m": mean("final_distance_m"),
        "mean_final_heading_error_deg": mean("final_heading_error_deg"),
        "mean_latency_ms": mean("latency_ms"),
    }
