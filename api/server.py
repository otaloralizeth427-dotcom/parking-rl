"""Servidor FastAPI que conecta el ambiente con la visualización 3D.

REST y WebSocket comparten el mismo despachador de comandos (SimulationSession.handle).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from agents.registry import list_manifests, load_manifest, load_policy
from agents.rule_based import RuleBasedPolicy
from envs.levels import LEVEL_IDS, env_kwargs, load_level, model_matches_level
from envs.parking_env import ACTION_NAMES, ParkingEnv

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
MAX_BATCH = 64            # pasos por petición «steps» (el cliente los anima uno a uno)
MAX_SESSIONS = 50         # partidas simultáneas en el servidor público
EXPERIMENTS_DIR = ROOT / "configs" / "experiments"
BASELINES = {
    "random": {"label": "Aleatoria", "mode": "Aleatorio"},
    "rule": {"label": "Regla", "mode": "Referencia (regla)"},
    "manual": {"label": "Manual (teclado)", "mode": "Manual"},
}


# Los modelos son de solo lectura: todas las partidas comparten la misma copia en memoria
_POLICIES: dict[str, object] = {}
_POLICIES_LOCK = threading.Lock()


def git_commit() -> str | None:
    """Commit del código servido (GIT_COMMIT o RENDER_GIT_COMMIT en el servidor, o git en local)."""
    for var in ("GIT_COMMIT", "RENDER_GIT_COMMIT"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class SimulationSession:
    def __init__(self, level: str | None = None, sensor_noise: float | None = None, public: bool = False):
        self._lock = threading.Lock()
        self.public = public          # versión publicada: sin entrenamiento desde la interfaz
        self.episode = 0
        self.collisions = 0
        self.successes = 0
        self.manual = False
        self.training = {"running": False, "episode": 0, "total": 0, "success_rate": 0.0, "returns": []}
        self._configure(level or LEVEL_IDS[0], sensor_noise)
        self.agent_name = self._default_agent()
        self.reset()

    # ----------------------------------------------------------- comandos
    def handle(self, cmd: str, payload: dict | None = None) -> dict:
        payload = payload or {}
        handlers = {
            "reset": lambda: self.reset(payload.get("seed"), payload.get("level"),
                                        payload.get("sensor_noise")),
            "step": lambda: self.step(payload.get("action")),
            "steps": lambda: self.steps(payload.get("n")),
            "state": self.state,
            "agent": lambda: self.set_agent(payload.get("agent")),
            "train": lambda: self.start_training(payload.get("episodes", 1000)),
            "config": self.config,
        }
        if cmd not in handlers:
            raise ValueError(f"Comando desconocido: {cmd!r}")
        return handlers[cmd]()

    def config(self) -> dict:
        return {"type": "config", "actions": ACTION_NAMES, "environment_version": self.environment_version,
                "levels": [{"id": lid, "name": load_level(lid)["name"]} for lid in LEVEL_IDS],
                "level": self.level_id, "sensor_noise": self.sensor_noise,
                "agent": self.agent_name, "agents": self.catalog(refresh=True),
                "can_train": not self.public and (EXPERIMENTS_DIR / f"{self.level_id}-exp01-dqn.json").exists()}

    def catalog(self, refresh: bool = False) -> list[dict]:
        """Referencias + modelos registrados del nivel actual (desde sus manifiestos).

        Se guarda por nivel; se vuelve a leer del disco al pedir la configuración (refresh=True),
        así un modelo recién entrenado aparece sin leer los manifiestos en cada paso.
        """
        if not refresh and getattr(self, "_catalog", None) and self._catalog[0] == self.level_id:
            return self._catalog[1]
        items = [{"id": key, **info, "available": True} for key, info in BASELINES.items()]
        for m in list_manifests(self.level_id):
            summary = m.get("evaluation_summary") or {}
            matches = model_matches_level(m, self.level_id)
            items.append({
                "id": m["run_id"], "label": f"{m['agent'].upper()} · {m['experiment_id']} · s{m['seed']}",
                "mode": "Evaluación (modelo congelado)", "available": m["status"] == "completed" and matches,
                "status": m["status"] if matches else "otra versión del nivel", "agent": m["agent"], "seed": m["seed"],
                "git_commit": m["git_commit"], "environment_version": m["environment_version"],
                "model_sha256": m.get("model_sha256"), "created_at": m["created_at"],
                "test_success_rate": summary.get("success_rate"), "test_mean_time_s": summary.get("mean_time_s"),
                "diagnostic": m.get("diagnostic"),
            })
        self._catalog = (self.level_id, items)
        return items

    def reset(self, seed=None, level=None, sensor_noise=None) -> dict:
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("seed debe ser un entero no negativo")
        with self._lock:
            if level is not None:
                self._configure(level, sensor_noise)
                if self.agent_name not in {a["id"] for a in self.catalog() if a["available"]}:
                    self.agent_name = self._default_agent()
            elif sensor_noise is not None:
                self._configure(self.level_id, sensor_noise)
            self.obs, _ = self.env.reset(seed=seed)
            self.episode += 1
            self.total_reward = 0.0
            self.last_reward = 0.0
            self.last_action = None
            self.manual = False
            self.status = "running"
            self.collision_kind = None
            return {"type": "reset", "layout": self.env.layout(), **self._frame()}

    def step(self, action=None) -> dict:
        with self._lock:
            if self.status != "running":
                raise ValueError("El episodio terminó; llama a reset")
            if action is None:
                if self.agent_name == "manual":
                    raise ValueError("En modo manual cada paso necesita una acción (flechas del teclado)")
                action = self._policy_action()
            elif not isinstance(action, int) or not self.env.action_space.contains(action):
                raise ValueError(f"Acción inválida: {action!r}. Usa 0-{self.env.action_space.n - 1}")
            else:
                self.manual = True
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            self.last_action = int(action)
            self.last_reward = reward
            self.total_reward += reward
            if info["collision"]:
                self.status = "collision"
                self.collision_kind = info["collision_kind"]
                self.collisions += 1
            elif info["success"]:
                self.status = "success"
                self.successes += 1
            elif truncated:
                self.status = "timeout"
            return {"type": "state", **self._frame()}

    def steps(self, n) -> dict:
        """Varios pasos del agente en una sola petición (evita esperar la red en cada paso).

        Se detiene al terminar el episodio; el cliente reproduce los cuadros a la velocidad elegida.
        """
        if not isinstance(n, int) or not 1 <= n <= MAX_BATCH:
            raise ValueError(f"n debe ser un entero entre 1 y {MAX_BATCH}")
        if self.agent_name == "manual":
            raise ValueError("En modo manual cada paso necesita una acción (flechas del teclado)")
        frames = []
        for _ in range(n):
            frames.append(self.step())
            if self.status != "running":
                break
        return {"type": "states", "frames": frames}

    def state(self) -> dict:
        with self._lock:
            return {"type": "state", **self._frame()}

    def set_agent(self, name) -> dict:
        available = {a["id"] for a in self.catalog() if a["available"]}
        if name not in available:
            raise ValueError(f"Agente no disponible en {self.level_id}: {name!r}")
        with self._lock:
            self._policy(name)          # valida el hash antes de activarlo
            self.agent_name = name
        return self.config()

    def start_training(self, episodes) -> dict:
        """Lanza el pipeline real: el run queda registrado en MLflow y con manifiesto."""
        if self.public:
            raise ValueError("El entrenamiento no está disponible en la versión publicada; se entrena en local")
        if not isinstance(episodes, int) or not 1 <= episodes <= 50_000:
            raise ValueError("episodes debe ser un entero entre 1 y 50000")
        if self.training["running"]:
            raise ValueError("Ya hay un entrenamiento en curso")
        base = EXPERIMENTS_DIR / f"{self.level_id}-exp01-dqn.json"
        if not base.exists():
            raise ValueError(f"No hay experimento base para {self.level_id}")
        from pipelines.train import load_experiment

        config = load_experiment(base)
        config.update({
            "experiment_id": f"{self.level_id}-ui-dqn",
            "hypothesis": f"Reentrenamiento lanzado desde la interfaz con {episodes} episodios.",
            "change_vs_baseline": f"Igual que {config['experiment_id']} cambiando el presupuesto a {episodes} episodios.",
            "seed": int(np.random.default_rng().integers(1_000_000)),
            "budget": {"episodes": episodes},
            "validation": {"episodes": 10, "every": max(1, episodes // 5)},
        })
        self.training = {"running": True, "episode": 0, "total": episodes, "success_rate": 0.0,
                         "returns": [], "experiment_id": config["experiment_id"]}
        threading.Thread(target=self._train_worker, args=(config,), daemon=True).start()
        return {"type": "training", "training": dict(self.training)}

    # ----------------------------------------------------------- internos
    def _configure(self, level_id: str, sensor_noise: float | None = None):
        """Usa el nivel versionado; sensor_noise permite sobrescribir su ruido (botón «Ruido»)."""
        level = load_level(level_id)
        kwargs = env_kwargs(level)
        if sensor_noise is not None:
            if float(sensor_noise) < 0:
                raise ValueError("sensor_noise debe ser >= 0")
            kwargs["sensor_noise"] = float(sensor_noise)
        self.env_kwargs = kwargs
        self.env = ParkingEnv(**kwargs)
        self.level_id, self.sensor_noise = level_id, kwargs["sensor_noise"]
        self.environment_version = level["environment_version"]
        self.status = "idle"

    def _default_agent(self) -> str:
        """El modelo con mayor éxito en la suite del nivel (a igual éxito, el más rápido); si no hay, la regla."""
        models = [a for a in self.catalog() if a["available"] and a["id"] not in BASELINES]
        best = lambda a: (a["test_success_rate"] or 0, -(a["test_mean_time_s"] or float("inf")))  # noqa: E731
        return max(models, key=best)["id"] if models else "rule"

    def _policy(self, name: str):
        if name in ("random", "manual"):
            return None
        with _POLICIES_LOCK:
            if name not in _POLICIES:
                _POLICIES[name] = RuleBasedPolicy() if name == "rule" else load_policy(load_manifest(name))
            return _POLICIES[name]

    def _policy_action(self) -> int:
        policy = self._policy(self.agent_name)
        return int(self.env.action_space.sample()) if policy is None else policy.act(self.obs, greedy=True)

    def _agent_info(self) -> dict:
        return next((a for a in self.catalog() if a["id"] == self.agent_name), {"label": self.agent_name})

    def _train_worker(self, config: dict):
        from pipelines.train import run_experiment

        def progress(episode, point):
            self.training["episode"] = episode
            self.training["success_rate"] = point["success_rate"]
            self.training["returns"] = (self.training["returns"] + [point["return"]])[-200:]

        try:
            manifest = run_experiment(config, progress=progress)
            self.training["run_id"] = manifest["run_id"]
        except Exception as exc:  # se reporta al cliente en lugar de morir en silencio
            self.training["error"] = str(exc)
        finally:
            self.training["running"] = False

    def _frame(self) -> dict:
        info = self._agent_info()
        return {
            **self.env.render_state(),
            "episode": self.episode,
            "level": self.level_id,
            "environment_version": self.environment_version,
            "collision_kind": self.collision_kind,
            "status": self.status,
            "reward": self.last_reward,
            "total_reward": self.total_reward,
            "action": self.last_action,
            "action_name": None if self.last_action is None else ACTION_NAMES[self.last_action],
            "agent": self.agent_name,
            "agent_label": info.get("label"),
            "mode": "Manual" if self.manual else info.get("mode"),
            "model": {k: info.get(k) for k in ("git_commit", "model_sha256", "seed", "test_success_rate")}
            if self.agent_name not in BASELINES else None,
            "collisions_total": self.collisions,
            "successes_total": self.successes,
            "training": dict(self.training),
        }


def create_app(session: SimulationSession | None = None, *, public: bool | None = None,
               allowed_origins: list[str] | None = None, allowed_origin_regex: str | None = None) -> FastAPI:
    """App de la simulación.

    - REST (/api/...) usa una partida compartida (scripts y pruebas).
    - Cada conexión WebSocket (/ws?level=N3) tiene su propia partida: varias personas pueden jugar a la vez.
    - public=True (o PARKING_PUBLIC=1) desactiva el entrenamiento desde la interfaz.
    - allowed_origins (o PARKING_ALLOWED_ORIGINS, separados por coma) habilita CORS para la web publicada;
      allowed_origin_regex (o PARKING_ALLOWED_ORIGIN_REGEX) admite también las vistas previas de Vercel.
    """
    if public is None:
        public = os.environ.get("PARKING_PUBLIC", "") == "1"
    if allowed_origins is None:
        allowed_origins = [o.strip() for o in os.environ.get("PARKING_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if allowed_origin_regex is None:
        allowed_origin_regex = os.environ.get("PARKING_ALLOWED_ORIGIN_REGEX") or None
    app = FastAPI(title="Parking RL")
    app.state.session = session or SimulationSession(os.environ.get("PARKING_LEVEL"), public=public)
    app.state.session.public = app.state.session.public or public
    app.state.default_level = app.state.session.level_id
    app.state.live_sessions = 0
    commit = git_commit()

    def call(cmd: str, payload: dict | None = None) -> dict:
        try:
            return app.state.session.handle(cmd, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def check_level(level: str | None) -> str:
        if level is None:
            return app.state.default_level
        if level not in LEVEL_IDS:
            raise HTTPException(status_code=400, detail=f"Nivel desconocido: {level!r}")
        return level

    @app.get("/api/health")
    def health():
        """Identidad de lo que está servido: commit del código y nivel por defecto."""
        return {"status": "ok", "git_commit": commit, "public": public, "default_level": app.state.default_level,
                "levels": list(LEVEL_IDS), "live_sessions": app.state.live_sessions}

    @app.get("/api/config")
    def get_config(level: str | None = None):
        if level is None or level == app.state.session.level_id:
            return call("config")
        # Solo para leer el catálogo de otro nivel (la web lo usa para elegir su paleta)
        return SimulationSession(check_level(level), public=public).config()

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

    def origin_allowed(origin: str | None) -> bool:
        # El navegador no aplica CORS a los WebSocket: se revisa aquí el mismo origen permitido
        if origin is None or not (allowed_origins or allowed_origin_regex):
            return True
        return origin in allowed_origins or bool(allowed_origin_regex and re.fullmatch(allowed_origin_regex, origin))

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        level = ws.query_params.get("level") or app.state.default_level
        if not origin_allowed(ws.headers.get("origin")):
            await ws.close(code=1008)
            return
        await ws.accept()
        if level not in LEVEL_IDS:
            await ws.close(code=1008, reason="Nivel desconocido")
            return
        if app.state.live_sessions >= MAX_SESSIONS:
            await ws.close(code=1013, reason="Servidor lleno, intenta en un momento")
            return
        app.state.live_sessions += 1
        base = app.state.session      # mismo ruido que se pidió al lanzar el servidor (--sensor-noise)
        try:
            # Crear la partida y calcular pasos va en un hilo aparte: así una partida no congela a las demás
            own = await run_in_threadpool(SimulationSession, level,
                                          base.sensor_noise if level == app.state.default_level else None,
                                          public=public)
            while True:
                try:
                    message = json.loads(await ws.receive_text())
                    if not isinstance(message, dict):
                        raise ValueError("El mensaje debe ser un objeto JSON")
                    response = await run_in_threadpool(own.handle, message.get("cmd", ""), message.get("payload"))
                except (ValueError, json.JSONDecodeError) as exc:
                    response = {"type": "error", "message": str(exc)}
                await ws.send_text(json.dumps(response))
        except WebSocketDisconnect:
            pass
        finally:
            app.state.live_sessions -= 1

    if allowed_origins or allowed_origin_regex:
        app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_origin_regex=allowed_origin_regex,
                           allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def no_cache(request, call_next):
        # Evita que el navegador use versiones viejas de la interfaz tras un cambio
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    # En local también se sirve la web (en producción la publica Vercel y aquí solo queda la API)
    if (WEB_DIR / "public").is_dir():
        app.mount("/models", StaticFiles(directory=WEB_DIR / "public" / "models"), name="models")
    if (WEB_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
