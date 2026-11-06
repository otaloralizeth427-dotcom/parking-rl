"""Servidor FastAPI que conecta el ambiente con la visualización 3D.

REST y WebSocket comparten el mismo despachador de comandos (SimulationSession.handle).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from agent_base import MODELS_DIR, Agent, train
from neural_q_learning import DQNAgent
from parking_env import ACTION_NAMES, DIFFICULTIES, ParkingEnv
from q_learning import QLearningAgent

AGENT_CLASSES: dict[str, type[Agent]] = {"q_learning": QLearningAgent, "neural": DQNAgent}
AGENT_NAMES = (*AGENT_CLASSES, "random")
TRAIN_DEFAULTS = {
    "q_learning": {"epsilon_decay": 0.9995},
    "neural": {"epsilon_decay": 0.998},
}
WEB_DIR = Path(__file__).parent / "web"


def history_path(agent_name: str) -> Path:
    return MODELS_DIR / f"{agent_name}_history.json"


def load_or_create(agent_name: str) -> tuple[Agent, bool]:
    """Carga el modelo guardado; si no existe devuelve un agente nuevo sin entrenar."""
    cls = AGENT_CLASSES[agent_name]
    try:
        return cls.load(), True
    except FileNotFoundError:
        return cls(seed=0, **TRAIN_DEFAULTS[agent_name]), False


class SimulationSession:
    def __init__(self, difficulty: str = "normal", sensor_noise: float = 0.0):
        self._lock = threading.Lock()
        self.agents: dict[str, Agent] = {}
        self.trained: dict[str, bool] = {}
        self.agent_name = "q_learning"
        self.episode = 0
        self.collisions = 0
        self.successes = 0
        self.training = {"running": False, "agent": None, "episode": 0, "total": 0,
                         "success_rate": 0.0, "returns": []}
        self._configure(difficulty, sensor_noise)
        self.reset()

    # ----------------------------------------------------------- comandos
    def handle(self, cmd: str, payload: dict | None = None) -> dict:
        payload = payload or {}
        handlers = {
            "reset": lambda: self.reset(payload.get("seed"), payload.get("difficulty"),
                                        payload.get("sensor_noise")),
            "step": lambda: self.step(payload.get("action")),
            "state": self.state,
            "agent": lambda: self.set_agent(payload.get("agent")),
            "train": lambda: self.start_training(payload.get("agent", self.agent_name),
                                                 payload.get("episodes", 1000)),
            "config": self.config,
        }
        if cmd not in handlers:
            raise ValueError(f"Comando desconocido: {cmd!r}")
        return handlers[cmd]()

    def config(self) -> dict:
        return {"type": "config", "agents": list(AGENT_NAMES), "difficulties": list(DIFFICULTIES),
                "actions": ACTION_NAMES, "difficulty": self.difficulty,
                "sensor_noise": self.sensor_noise, "agent": self.agent_name,
                "trained": {n: self._is_trained(n) for n in AGENT_NAMES}}

    def reset(self, seed=None, difficulty=None, sensor_noise=None) -> dict:
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("seed debe ser un entero no negativo")
        with self._lock:
            if difficulty is not None or sensor_noise is not None:
                self._configure(difficulty or self.difficulty,
                                self.sensor_noise if sensor_noise is None else float(sensor_noise))
            self.obs, _ = self.env.reset(seed=seed)
            self.episode += 1
            self.total_reward = 0.0
            self.last_reward = 0.0
            self.last_action = None
            self.status = "running"
            return {"type": "reset", "layout": self.env.layout(), **self._frame()}

    def step(self, action=None) -> dict:
        with self._lock:
            if self.status != "running":
                raise ValueError("El episodio terminó; llama a reset")
            if action is None:
                action = self._policy_action()
            elif not isinstance(action, int) or not self.env.action_space.contains(action):
                raise ValueError(f"Acción inválida: {action!r}. Usa 0-{self.env.action_space.n - 1}")
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            self.last_action = int(action)
            self.last_reward = reward
            self.total_reward += reward
            if info["collision"]:
                self.status = "collision"
                self.collisions += 1
            elif info["success"]:
                self.status = "success"
                self.successes += 1
            elif truncated:
                self.status = "timeout"
            return {"type": "state", **self._frame()}

    def state(self) -> dict:
        with self._lock:
            return {"type": "state", **self._frame()}

    def set_agent(self, name) -> dict:
        if name not in AGENT_NAMES:
            raise ValueError(f"Agente desconocido: {name!r}. Opciones: {AGENT_NAMES}")
        with self._lock:
            self.agent_name = name
        return self.config()

    def start_training(self, agent_name, episodes) -> dict:
        if agent_name not in AGENT_CLASSES:
            raise ValueError(f"Solo se pueden entrenar: {list(AGENT_CLASSES)}")
        if not isinstance(episodes, int) or not 1 <= episodes <= 50_000:
            raise ValueError("episodes debe ser un entero entre 1 y 50000")
        if self.training["running"]:
            raise ValueError("Ya hay un entrenamiento en curso")
        agent = self._agent(agent_name)
        self.training = {"running": True, "agent": agent_name, "episode": 0, "total": episodes,
                         "success_rate": 0.0, "returns": []}
        threading.Thread(target=self._train_worker, args=(agent_name, agent, episodes),
                         daemon=True).start()
        return {"type": "training", "training": dict(self.training)}

    # ----------------------------------------------------------- internos
    def _configure(self, difficulty: str, sensor_noise: float):
        self.env = ParkingEnv(difficulty, sensor_noise)
        self.difficulty, self.sensor_noise = difficulty, sensor_noise
        self.status = "idle"

    def _agent(self, name: str) -> Agent:
        if name not in self.agents:
            self.agents[name], self.trained[name] = load_or_create(name)
        return self.agents[name]

    def _is_trained(self, name: str) -> bool:
        if name == "random":
            return True
        if name in self.trained:
            return self.trained[name]
        return AGENT_CLASSES[name].default_path.exists()

    def _policy_action(self) -> int:
        if self.agent_name == "random":
            return int(self.env.action_space.sample())
        return self._agent(self.agent_name).act(self.obs, greedy=True)

    def _train_worker(self, name: str, agent: Agent, episodes: int):
        env = ParkingEnv(self.difficulty, self.sensor_noise)
        successes: list[bool] = []

        def progress(ep, result):
            successes.append(result["success"])
            self.training["episode"] = ep + 1
            self.training["success_rate"] = float(np.mean(successes[-100:]))
            self.training["returns"] = (self.training["returns"] + [round(result["return"], 2)])[-200:]

        try:
            seed = int(np.random.default_rng().integers(1_000_000))
            history = train(agent, env, episodes, seed=seed, callback=progress)
            agent.save()
            path = history_path(name)
            previous = json.loads(path.read_text()) if path.exists() else []
            path.write_text(json.dumps(previous + history))
            self.trained[name] = True
        except Exception as exc:  # se reporta al cliente en lugar de morir en silencio
            self.training["error"] = str(exc)
        finally:
            self.training["running"] = False

    def _frame(self) -> dict:
        return {
            **self.env.render_state(),
            "episode": self.episode,
            "status": self.status,
            "reward": self.last_reward,
            "total_reward": self.total_reward,
            "action": self.last_action,
            "action_name": None if self.last_action is None else ACTION_NAMES[self.last_action],
            "agent": self.agent_name,
            "agent_trained": self._is_trained(self.agent_name),
            "collisions_total": self.collisions,
            "successes_total": self.successes,
            "training": dict(self.training),
        }


def create_app(session: SimulationSession | None = None) -> FastAPI:
    app = FastAPI(title="Parking RL")
    app.state.session = session or SimulationSession()

    def call(cmd: str, payload: dict | None = None) -> dict:
        try:
            return app.state.session.handle(cmd, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/config")
    def get_config():
        return call("config")

    @app.get("/api/state")
    def get_state():
        return call("state")

    @app.post("/api/reset")
    def post_reset(payload: dict | None = None):
        return call("reset", payload)

    @app.post("/api/step")
    def post_step(payload: dict | None = None):
        return call("step", payload)

    @app.post("/api/agent")
    def post_agent(payload: dict):
        return call("agent", payload)

    @app.post("/api/train")
    def post_train(payload: dict):
        return call("train", payload)

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                try:
                    message = json.loads(await ws.receive_text())
                    if not isinstance(message, dict):
                        raise ValueError("El mensaje debe ser un objeto JSON")
                    response = app.state.session.handle(message.get("cmd", ""), message.get("payload"))
                except (ValueError, json.JSONDecodeError) as exc:
                    response = {"type": "error", "message": str(exc)}
                await ws.send_text(json.dumps(response))
        except WebSocketDisconnect:
            pass

    @app.middleware("http")
    async def no_cache(request, call_next):
        # Evita que el navegador use versiones viejas de la interfaz tras un cambio
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
