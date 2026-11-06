"""Catálogo de modelos registrados (manifiestos) y carga verificada por hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agents.neural_q_learning import DQNAgent
from agents.q_learning import QLearningAgent

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS_DIR = ROOT / "manifests"
ARTIFACTS_DIR = ROOT / "artifacts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_manifests(level_id: str | None = None) -> list[dict]:
    """Manifiestos de modelos (manifests/*.json), del más antiguo al más reciente."""
    manifests = [json.loads(p.read_text()) for p in MANIFESTS_DIR.glob("*.json")]
    if level_id:
        manifests = [m for m in manifests if m["level_id"] == level_id]
    return sorted(manifests, key=lambda m: m["created_at"])


def load_manifest(run_id: str) -> dict:
    path = MANIFESTS_DIR / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe el manifiesto del run {run_id}")
    return json.loads(path.read_text())


def load_policy(manifest: dict):
    """Carga el modelo de un manifiesto, comprobando que su hash coincide con el registrado."""
    from agents.ppo_agent import PPOAgent

    if manifest["status"] != "completed" or not manifest.get("model_uri"):
        raise ValueError(f"El run {manifest['run_id']} no tiene checkpoint disponible")
    path = ROOT / manifest["model_uri"]
    if not path.exists():
        raise FileNotFoundError(f"Falta el artefacto {manifest['model_uri']}")
    actual = sha256_file(path)
    if actual != manifest["model_sha256"]:
        raise ValueError(f"El hash de {manifest['model_uri']} no coincide con el manifiesto "
                         f"({actual[:12]}… ≠ {manifest['model_sha256'][:12]}…)")
    loaders = {"dqn": DQNAgent.load, "q_learning": QLearningAgent.load, "ppo": PPOAgent.load}
    return loaders[manifest["agent"]](path)
