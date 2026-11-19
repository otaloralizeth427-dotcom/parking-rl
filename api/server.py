"""Servidor FastAPI que conecta el ambiente con la visualización 3D.

REST y WebSocket comparten el mismo despachador de comandos (SimulationSession.handle).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from agents.registry import list_manifests, load_manifest, load_policy
from agents.rule_based import RuleBasedPolicy
from agents.serving import load_serving, previous_run, served_run
from envs.levels import WEB_LEVEL_IDS as LEVEL_IDS, env_kwargs, load_level, model_matches_level
from envs.parking_env import ACTION_NAMES, ParkingEnv

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
MAX_BATCH = 64            # pasos por petición «steps» (el cliente los anima uno a uno)
MAX_SESSIONS = 50         # partidas simultáneas en el servidor público
EXPERIMENTS_DIR = ROOT / "configs" / "experiments"
RESULTS_DIR = ROOT / "results"
BASELINES = {
    "random": {"label": "Aleatoria", "mode": "Aleatorio"},
    "rule": {"label": "Regla", "mode": "Referencia (regla)"},
    "manual": {"label": "Manual (teclado)", "mode": "Manual"},
}


MAX_EVAL_EPISODES = 30   # evaluación desde la web: primeros casos de la suite fija del nivel
MAX_EVAL_AGENTS = 4


class Metrics:
    """Monitoreo del servidor: resultados por nivel y agente, causas de fallo y latencia por paso."""

    def __init__(self):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.episodes: dict[tuple[str, str], Counter] = {}
        self.latency_ms: deque[float] = deque(maxlen=5000)
        self.steps = 0

    def step(self, ms: float):
        with self._lock:
            self.steps += 1
            self.latency_ms.append(ms)

    def episode(self, level_id: str, agent: str, outcome: str):
        with self._lock:
            self.episodes.setdefault((level_id, agent), Counter())[outcome] += 1

    def snapshot(self) -> dict:
        with self._lock:
            lat = np.array(self.latency_ms) if self.latency_ms else np.zeros(1)
            by_level: dict[str, dict] = {}
            for (level_id, agent), counts in self.episodes.items():
                level = by_level.setdefault(level_id, {"episodes": 0, "outcomes": Counter(), "agents": {}})
                n = sum(counts.values())
                level["episodes"] += n
                level["outcomes"].update(counts)
                level["agents"][agent] = {"episodes": n, "success_rate": round(counts["success"] / n, 4),
                                          "outcomes": dict(counts)}
            for level in by_level.values():
                level["success_rate"] = round(level["outcomes"]["success"] / level["episodes"], 4)
                level["outcomes"] = dict(level["outcomes"])
            return {"uptime_s": round(time.time() - self.started_at), "steps": self.steps,
                    "step_latency_ms": {"mean": round(float(lat.mean()), 3), "p95": round(float(np.percentile(lat, 95)), 3),
                                        "window": len(self.latency_ms)},
                    "levels": by_level}


METRICS = Metrics()

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
            "evaluate": lambda: self.evaluate(payload.get("agents"), payload.get("episodes", 10)),
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
                "agent": self.agent_name, "agents": self.catalog(refresh=True), "serving": self.serving(),
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
            experiment = self._experiment_metadata(m)
            items.append({
                "id": m["run_id"], "label": f"{m['agent'].upper()} · {m['experiment_id']} · s{m['seed']}",
                "mode": "Evaluación (modelo congelado)", "available": m["status"] == "completed" and matches,
                "status": m["status"] if matches else "otra versión del nivel", "agent": m["agent"], "seed": m["seed"],
                "git_commit": m["git_commit"], "environment_version": m["environment_version"],
                "model_sha256": m.get("model_sha256"), "created_at": m["created_at"],
                "test_success_rate": summary.get("success_rate"), "test_mean_time_s": summary.get("mean_time_s"),
                "diagnostic": m.get("diagnostic"), "hypothesis": m.get("hypothesis"),
                "change_vs_baseline": m.get("change_vs_baseline"), "budget": m.get("budget"),
                "parent_run_id": m.get("parent_run_id"), "evaluation_suite_id": m.get("evaluation_suite_id"),
                "outcomes": summary.get("outcomes", {}), "mean_return": summary.get("mean_return"),
                "mean_final_distance_m": summary.get("mean_final_distance_m"),
                "mean_final_heading_error_deg": summary.get("mean_final_heading_error_deg"),
                "mlflow": m.get("mlflow"), "experiment": experiment,
                "mlflow_run_id": (m.get("mlflow") or {}).get("run_id"),
                "mlflow_experiment": (m.get("mlflow") or {}).get("experiment"),
            })
        self._catalog = (self.level_id, items)
        return items

    def serving(self) -> dict | None:
        """Modelo servido del nivel (puntero versionado) y el anterior, para el rollback desde la web."""
        serving = load_serving(self.level_id)
        if not serving:
            return None
        return {"current": serving["current"], "previous": previous_run(self.level_id),
                "history": serving["history"][-10:]}

    def evaluate(self, agents, episodes) -> dict:
        """Evalúa uno o varios agentes con los mismos primeros casos de la suite fija del nivel.

        Mismo código que el pipeline (pipelines/scoring.py), así lo que se ve en la web es comparable.
        """
        from pipelines.scoring import run_case, suite_for_level, summarize

        if not isinstance(agents, list) or not 1 <= len(agents) <= MAX_EVAL_AGENTS:
            raise ValueError(f"agents debe ser una lista de 1 a {MAX_EVAL_AGENTS} agentes")
        if not isinstance(episodes, int) or not 1 <= episodes <= MAX_EVAL_EPISODES:
            raise ValueError(f"episodes debe ser un entero entre 1 y {MAX_EVAL_EPISODES}")
        available = {a["id"]: a for a in self.catalog() if a["available"] and a["id"] != "manual"}
        unknown = [a for a in agents if a not in available]
        if unknown:
            raise ValueError(f"No se pueden evaluar en {self.level_id}: {unknown}")
        suite = suite_for_level(self.level_id)
        env = ParkingEnv(**env_kwargs(load_level(self.level_id)))      # sin el ruido extra del botón
        results = []
        for name in agents:
            cases = [run_case(env, self._policy(name), suite["seed_start"] + i) for i in range(episodes)]
            info = available[name]
            results.append({"agent": name, "label": info["label"], "environment_version": self.environment_version,
                            "model_sha256": info.get("model_sha256"), "git_commit": info.get("git_commit"),
                            "summary": summarize(cases), "outcomes_by_case": [c["outcome"] for c in cases]})
        return {"type": "evaluation", "level": self.level_id, "suite_id": suite["suite_id"],
                "seed_start": suite["seed_start"], "episodes": episodes, "results": results}

    @staticmethod
    def _experiment_metadata(manifest: dict) -> dict | None:
        path = EXPERIMENTS_DIR / f"{manifest.get('experiment_id')}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

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
            started = time.perf_counter()
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            METRICS.step((time.perf_counter() - started) * 1000)
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
            if self.status != "running" and not self.manual:
                METRICS.episode(self.level_id, self.agent_name,
                                "success" if self.status == "success" else self.collision_kind or "timeout")
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
        """El modelo servido del nivel (serving/<nivel>.json). Si no hay puntero: el de mayor éxito en la
        suite (a igual éxito, el más rápido); si no hay modelos, la regla."""
        models = [a for a in self.catalog() if a["available"] and a["id"] not in BASELINES]
        served = served_run(self.level_id)
        if served in {a["id"] for a in models}:
            return served
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
            "model": {k: info.get(k) for k in ("git_commit", "model_sha256", "seed", "test_success_rate",
                                                "mlflow_run_id", "mlflow_experiment")}
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
                "levels": list(LEVEL_IDS), "live_sessions": app.state.live_sessions,
                "served": {lid: served_run(lid) for lid in LEVEL_IDS}}

    @app.get("/api/serving")
    def get_serving(level: str | None = None):
        level = check_level(level)
        return load_serving(level) or {"level_id": level, "current": None, "history": []}

    @app.get("/api/metrics")
    def get_metrics():
        """Monitoreo: éxito y causas de fallo por nivel/agente, latencia por paso y versión servida."""
        return {**METRICS.snapshot(), "git_commit": commit, "live_sessions": app.state.live_sessions,
                "served": {lid: served_run(lid) for lid in LEVEL_IDS}}

    @app.get("/api/levels")
    def get_levels():
        """Los niveles con su nombre, versión del ambiente y el éxito del modelo servido (para el selector)."""
        out = []
        for lid in LEVEL_IDS:
            level = load_level(lid)
            served = served_run(lid)
            summary = (load_manifest(served).get("evaluation_summary") or {}) if served else {}
            out.append({"id": lid, "name": level["name"], "environment_version": level["environment_version"],
                        "served": served, "success_rate": summary.get("success_rate")})
        return out

    @app.get("/api/level-metrics")
    def get_level_metrics(level: str | None = None):
        """Métricas de un nivel: modelo servido (identidad), suite fija y test final reservado.

        Sale de los resultados versionados del pipeline (results/<nivel>/), no de partidas en vivo.
        """
        level = check_level(level)
        info = load_level(level)
        served = served_run(level)
        manifest = load_manifest(served) if served else None
        strip = lambda p: {k: v for k, v in p.items() if k != "outcomes_by_case"}  # noqa: E731

        def results(name: str) -> dict | None:
            path = RESULTS_DIR / level / name
            if not path.exists():
                return None
            data = json.loads(path.read_text())
            return {**{k: v for k, v in data.items() if k != "policies"},
                    "policies": {key: strip(p) for key, p in data["policies"].items()}}

        return {
            "level": level, "name": info["name"], "environment_version": info["environment_version"],
            "description": info.get("description"),
            "served": None if manifest is None else {
                k: manifest.get(k) for k in ("run_id", "agent", "experiment_id", "seed", "hypothesis", "git_commit",
                                             "model_sha256", "environment_version", "evaluation_suite_id",
                                             "parent_run_id", "created_at", "budget")} | {
                "mlflow_run_id": (manifest.get("mlflow") or {}).get("run_id"),
                "mlflow_experiment": (manifest.get("mlflow") or {}).get("experiment")},
            "suite": results("comparison.json"),
            "final_test": results("final_test.json"),
            "has_curves": (RESULTS_DIR / level / "curves.png").exists(),
        }

    @app.get("/api/level-metrics/curves.png", include_in_schema=False)
    def get_level_curves(level: str | None = None):
        from fastapi.responses import FileResponse
        path = RESULTS_DIR / check_level(level) / "curves.png"
        if not path.exists():
            raise HTTPException(status_code=404, detail="El nivel no tiene curvas de entrenamiento")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/config")
    def get_config(level: str | None = None):
        if level is None or level == app.state.session.level_id:
            return call("config")
        # Solo para leer el catálogo de otro nivel (la web lo usa para elegir su paleta)
        return SimulationSession(check_level(level), public=public).config()

    @app.get("/api/catalog")
    def get_catalog(level: str | None = None):
        """Catálogo de modelos de un nivel (por defecto, el de la partida REST) con su metadata."""
        session = app.state.session
        if level is not None and check_level(level) != session.level_id:
            session = SimulationSession(level, public=public)
        return {"level": session.level_id, "environment_version": session.environment_version,
                "recommended": session._default_agent(), "items": session.catalog()}

    @app.get("/api/model/{run_id}/evaluation")
    def get_model_evaluation(run_id: str):
        try:
            manifest = load_manifest(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        uri = manifest.get("evaluation_uri")
        if not uri:
            raise HTTPException(status_code=404, detail="El modelo no tiene evaluación guardada")
        path = ROOT / uri
        if not path.exists():
            raise HTTPException(status_code=404, detail="Falta el artefacto de evaluación")
        return {"run_id": run_id, "manifest": manifest, "evaluation": json.loads(path.read_text())}

    @app.post("/api/rollback")
    def rollback(payload: dict):
        """Activa un modelo histórico compatible en la partida REST y reinicia el episodio.

        Es reversible: cualquier modelo anterior puede volver a activarse por su run_id. Cada visitante de
        la web lo hace en su propia partida (comando «agent» del WebSocket); el rollback del modelo servido
        a todos es el puntero versionado serving/<nivel>.json (python -m agents.serving rollback).
        """
        run_id = payload.get("run_id") if isinstance(payload, dict) else None
        if not run_id:
            raise HTTPException(status_code=400, detail="run_id es obligatorio")
        config = call("agent", {"agent": run_id})
        reset = call("reset", {})
        return {"type": "rollback", "run_id": run_id, "config": config, "reset": reset}

    @app.get("/api/compare")
    def compare(run_ids: str = Query(..., description="run_id separados por coma")):
        wanted = [value.strip() for value in run_ids.split(",") if value.strip()]
        if not 2 <= len(wanted) <= 8:
            raise HTTPException(status_code=400, detail="Selecciona entre 2 y 8 modelos")
        by_id = {item["id"]: item for item in app.state.session.catalog()}
        missing = [run_id for run_id in wanted if run_id not in by_id or not by_id[run_id]["available"]]
        if missing:
            raise HTTPException(status_code=409, detail=f"Modelos no compatibles: {', '.join(missing)}")
        return {"level": app.state.session.level_id, "items": [by_id[run_id] for run_id in wanted]}

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

    # En local también se sirve la web ya construida (cd web && npm ci && npm run build).
    # En producción la publica Vercel y aquí solo queda la API.
    dist = WEB_DIR / "dist"
    if (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    elif WEB_DIR.exists():
        @app.get("/", include_in_schema=False)
        def web_not_built():
            from fastapi.responses import HTMLResponse
            return HTMLResponse("<h1>Falta construir la web</h1><p>Ejecute <code>cd web && npm ci && npm run build</code> "
                                "y recargue. La API ya está disponible en <code>/api</code>.</p>", status_code=503)
    return app
