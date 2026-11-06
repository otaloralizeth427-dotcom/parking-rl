"""Catálogo de agentes disponibles y utilidades para cargarlos."""

from __future__ import annotations

from pathlib import Path

from agents.agent_base import MODELS_DIR, Agent
from agents.neural_q_learning import DQNAgent
from agents.q_learning import QLearningAgent

AGENT_CLASSES: dict[str, type[Agent]] = {"q_learning": QLearningAgent, "neural": DQNAgent}
AGENT_NAMES = (*AGENT_CLASSES, "random")
TRAIN_DEFAULTS = {
    "q_learning": {"epsilon_decay": 0.9995},
    "neural": {"epsilon_decay": 0.998},
}


def history_path(agent_name: str) -> Path:
    return MODELS_DIR / f"{agent_name}_history.json"


def load_or_create(agent_name: str) -> tuple[Agent, bool]:
    """Carga el modelo guardado; si no existe devuelve un agente nuevo sin entrenar."""
    cls = AGENT_CLASSES[agent_name]
    try:
        return cls.load(), True
    except FileNotFoundError:
        return cls(seed=0, **TRAIN_DEFAULTS[agent_name]), False
