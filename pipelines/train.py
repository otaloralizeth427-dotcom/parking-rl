"""Entrena un experimento versionado, lo registra en MLflow y empaqueta el modelo con su manifiesto.

Uso:  python -m pipelines.train configs/experiments/N1-exp01-dqn.json [--seed 43]
--seed repite el mismo experimento con otra semilla (réplicas para medir la variación).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agents.registry import ARTIFACTS_DIR, MANIFESTS_DIR, load_manifest, sha256_file
from envs.levels import env_kwargs, load_level, make_env
from envs.parking_env import ENV_VERSION
from pipelines.evaluate import evaluate, run_case, suite_for_level

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

ROOT = Path(__file__).resolve().parents[1]
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACTS = ROOT / "mlartifacts"
AGENTS = ("dqn", "q_learning", "ppo")
REQUIRED_KEYS = ("experiment_id", "level_id", "agent", "hypothesis", "change_vs_baseline", "seed", "budget")
VALIDATION_SEED_START = 200_000   # semillas de validación: distintas de entrenamiento y de la suite
SUCCESS_WINDOW = 100


# ------------------------------------------------------------------ configuración
def load_experiment(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ValueError(f"Faltan claves en {path}: {missing}")
    if config["agent"] not in AGENTS:
        raise ValueError(f"agent debe ser uno de {AGENTS}")
    return config


def resolve(config: dict) -> dict:
    """Configuración completa e inmutable del run: experimento + nivel + suite + versión."""
    budget = config["budget"]
    unit = "timesteps" if config["agent"] == "ppo" else "episodes"
    if unit not in budget or budget[unit] <= 0:
        raise ValueError(f"El agente {config['agent']} necesita budget.{unit} > 0")
    return {
        **config,
        "environment_version": ENV_VERSION,
        "level": load_level(config["level_id"]),
        "evaluation_suite": suite_for_level(config["level_id"]),
        "validation": {"seed_start": VALIDATION_SEED_START,
                       "episodes": config.get("validation", {}).get("episodes", 20),
                       "every": config.get("validation", {}).get("every", budget[unit] // 20)},
        "hyperparameters": config.get("hyperparameters", {}),
    }


def config_hash(resolved: dict) -> str:
    return hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()


def git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--", "envs", "agents", "pipelines", "configs"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "branch": None, "dirty": True}


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


# ------------------------------------------------------------------ curvas
class Curves:
    """Curva de entrenamiento (con exploración) y de validación (política congelada), separadas."""

    def __init__(self, resolved: dict, mlflow, policy_fn, progress=None):
        self.mlflow, self.policy_fn, self.progress = mlflow, policy_fn, progress
        self.validation = resolved["validation"]
        self.level_id = resolved["level_id"]
        self.train, self.val, self._successes = [], [], []
        self._next_val = self.validation["every"]

    def episode(self, step: int, result: dict):
        self._successes.append(result["success"])
        point = {"step": step, "return": round(result["return"], 3), "success": bool(result["success"]),
                 "success_rate": float(np.mean(self._successes[-SUCCESS_WINDOW:]))}
        self.train.append(point)
        if self.progress:
            self.progress(len(self.train), point)
        if len(self.train) % 10 == 0:
            self.mlflow.log_metrics({"train_success_rate": point["success_rate"],
                                     "train_return": point["return"]}, step=step)
        if step >= self._next_val:
            self._next_val += self.validation["every"]
            self.validate(step)

    def validate(self, step: int):
        env, policy = make_env(self.level_id), self.policy_fn()
        cases = [run_case(env, policy, self.validation["seed_start"] + i) for i in range(self.validation["episodes"])]
        rate = float(np.mean([c["success"] for c in cases]))
        self.val.append({"step": step, "success_rate": rate,
                         "mean_return": float(np.mean([c["return"] for c in cases]))})
        self.mlflow.log_metric("val_success_rate", rate, step=step)


def parent_policy(resolved: dict):
    """Checkpoint del que parte un run (verificado por hash). Debe compartir agente, ambiente y contratos."""
    from agents.registry import load_manifest, load_policy

    parent = load_manifest(resolved["parent_run_id"])
    if parent["agent"] != resolved["agent"]:
        raise ValueError(f"El padre es {parent['agent']}, no {resolved['agent']}")
    if parent["environment_version"] != ENV_VERSION:
        raise ValueError(f"El padre usa {parent['environment_version']}; las observaciones podrían no coincidir")
    return load_policy(parent)


# ------------------------------------------------------------------ entrenamiento por algoritmo
def train_tabular_or_dqn(resolved: dict, curves_factory):
    from agents.agent_base import train
    from agents.neural_q_learning import DQNAgent
    from agents.q_learning import QLearningAgent

    cls = DQNAgent if resolved["agent"] == "dqn" else QLearningAgent
    agent = cls(seed=resolved["seed"], **resolved["hyperparameters"])
    curves = curves_factory(lambda: agent)
    env = make_env(resolved["level_id"])
    env_steps = 0

    def on_episode(ep, result):
        nonlocal env_steps
        env_steps += result["steps"]
        curves.episode(ep + 1, result)

    train(agent, env, resolved["budget"]["episodes"], seed=resolved["seed"], callback=on_episode)
    return agent, curves, {"episodes": resolved["budget"]["episodes"], "env_steps": env_steps}


def train_ppo(resolved: dict, curves_factory):
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    from agents.ppo_agent import PPOAgent

    env = Monitor(make_env(resolved["level_id"]), info_keywords=("success",))
    env.reset(seed=resolved["seed"])
    if resolved.get("parent_run_id"):
        agent = PPOAgent.continue_from(parent_policy(resolved), env, seed=resolved["seed"],
                                       **resolved["hyperparameters"])
    else:
        agent = PPOAgent.create(env, seed=resolved["seed"], **resolved["hyperparameters"])
    curves = curves_factory(lambda: agent)
    episodes = 0

    class CurveCallback(BaseCallback):
        def _on_step(self) -> bool:
            nonlocal episodes
            for info in self.locals["infos"]:
                if "episode" in info:
                    episodes += 1
                    curves.episode(self.num_timesteps, {"return": info["episode"]["r"],
                                                        "success": info.get("success", False)})
            return True

    agent.model.learn(total_timesteps=resolved["budget"]["timesteps"], callback=CurveCallback())
    return agent, curves, {"timesteps": resolved["budget"]["timesteps"], "episodes": episodes,
                           "env_steps": resolved["budget"]["timesteps"]}


# ------------------------------------------------------------------ run completo
def run_experiment(config: dict, progress=None) -> dict:
    import mlflow

    resolved = resolve(config)
    cfg_sha = config_hash(resolved)
    git = git_state()
    if git["dirty"]:
        print("Aviso: hay cambios sin commit en el código; el manifiesto lo registrará (git_dirty).")

    mlflow.set_tracking_uri(MLFLOW_URI)
    experiment_name = f"parking-{resolved['level_id']}"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{resolved['experiment_id']}-seed{resolved['seed']}") as active:
        run_id = f"{resolved['experiment_id']}-s{resolved['seed']}-{active.info.run_id[:8]}"
        out_dir = ARTIFACTS_DIR / resolved["level_id"] / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(resolved, indent=2, ensure_ascii=False))
        mlflow.log_params({k: str(v)[:250] for k, v in flatten(
            {k: resolved[k] for k in ("agent", "seed", "budget", "hyperparameters", "level_id")}).items()})
        mlflow.set_tags({"run_id": run_id, "git_commit": git["commit"], "git_branch": git["branch"],
                         "git_dirty": git["dirty"], "environment_version": ENV_VERSION,
                         "level_id": resolved["level_id"], "hypothesis": resolved["hypothesis"],
                         "change_vs_baseline": resolved["change_vs_baseline"], "config_sha256": cfg_sha,
                         "parent_run_id": resolved.get("parent_run_id") or "none"})
        manifest = {
            "run_id": run_id, "experiment_id": resolved["experiment_id"], "level_id": resolved["level_id"],
            "agent": resolved["agent"], "hypothesis": resolved["hypothesis"],
            "change_vs_baseline": resolved["change_vs_baseline"],
            "environment_version": ENV_VERSION, "environment_kwargs": env_kwargs(resolved["level"]),
            "git_commit": git["commit"], "git_branch": git["branch"], "git_dirty": git["dirty"],
            "config_uri": str((out_dir / "config.json").relative_to(ROOT)), "config_sha256": cfg_sha,
            "seed": resolved["seed"], "parent_run_id": resolved.get("parent_run_id"),
            "origin": "continuación de checkpoint" if resolved.get("parent_run_id") else "desde cero",
            "parent_model_sha256": (load_manifest(resolved["parent_run_id"])["model_sha256"]
                                    if resolved.get("parent_run_id") else None),
            "evaluation_suite_id": resolved["evaluation_suite"]["suite_id"],
            "mlflow": {"tracking_uri": "sqlite:///mlflow.db", "experiment": experiment_name,
                       "run_id": active.info.run_id},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        started = time.time()
        try:
            curves_factory = lambda policy_fn: Curves(resolved, mlflow, policy_fn, progress)  # noqa: E731
            trainer = train_ppo if resolved["agent"] == "ppo" else train_tabular_or_dqn
            agent, curves, budget = trainer(resolved, curves_factory)
            wall = round(time.time() - started, 1)

            model_path = agent.save(out_dir / "model")
            if not model_path.exists():          # DQN/Q guardan .npz; PPO guarda .zip
                model_path = next(out_dir.glob("model.*"))
            (out_dir / "curves.json").write_text(json.dumps({"train": curves.train, "validation": curves.val}))
            evaluation = evaluate(agent, resolved["evaluation_suite"])
            (out_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=1))
            summary = evaluation["summary"]
            mlflow.log_metrics({"test_success_rate": summary["success_rate"],
                                "test_mean_return": summary["mean_return"],
                                "test_mean_time_s": summary["mean_time_s"],
                                "test_mean_latency_ms": summary["mean_latency_ms"],
                                "wall_time_s": wall, "env_steps": budget["env_steps"]})
            manifest.update({
                "status": "completed",
                "model_uri": str(model_path.relative_to(ROOT)), "model_sha256": sha256_file(model_path),
                "normalization_uri": "envs/parking_env.py#ParkingEnv._observation "
                                     "(escalado fijo dentro del ambiente; sin normalización aprendida)",
                "observation_contract": "Box(14,) float32 · ver docs/ficha_ambiente.md",
                "action_contract": "Discrete(5): avanzar, retroceder, izquierda, derecha, frenar",
                "runtime_ref": f"python {platform.python_version()} · requirements.lock.txt "
                               f"sha256:{sha256_file(ROOT / 'requirements.lock.txt')[:16]}",
                "budget": {**budget, "wall_time_s": wall},
                "optimizer_state": "no guardado: los runs no se continúan desde este checkpoint",
                "evaluation_uri": str((out_dir / "evaluation.json").relative_to(ROOT)),
                "evaluation_summary": summary,
            })
        except Exception as exc:
            manifest.update({"status": "failed", "model_uri": None, "model_sha256": None,
                             "checkpoint": "sin checkpoint disponible",
                             "diagnostic": f"{type(exc).__name__}: {exc}",
                             "budget": {"wall_time_s": round(time.time() - started, 1)}})
            (out_dir / "error.txt").write_text(traceback.format_exc())
            mlflow.set_tag("diagnostic", manifest["diagnostic"])
            raise
        finally:
            (MANIFESTS_DIR / f"{run_id}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
            shutil.copy(MANIFESTS_DIR / f"{run_id}.json", out_dir / "manifest.json")
            mlflow.log_artifacts(str(out_dir), artifact_path="package")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="archivo configs/experiments/*.json")
    parser.add_argument("--seed", type=int, help="sobrescribe la semilla del experimento")
    args = parser.parse_args(argv)
    config = load_experiment(args.experiment)
    if args.seed is not None:
        config["seed"] = args.seed
    manifest = run_experiment(config)
    s = manifest["evaluation_summary"]
    print(f"{manifest['run_id']}: éxito {s['success_rate']:.0%} en {manifest['evaluation_suite_id']} "
          f"· manifiesto en manifests/{manifest['run_id']}.json")


if __name__ == "__main__":
    main()
