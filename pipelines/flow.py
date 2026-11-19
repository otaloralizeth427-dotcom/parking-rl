"""Pipeline completo con Prefect: de una configuración (o un checkpoint) hasta el modelo servido.

    python -m pipelines.flow --level N5 --run-id <run_id>                   # candidato ya entrenado
    python -m pipelines.flow --level N5 --experiment configs/experiments/N5-exp05-ppo-inverso-3m.json --seed 45
    python -m pipelines.flow ... --dry-run    # decide sin mover el puntero del modelo servido

Pasos:
  1. validar
  2. pruebas
  3. entrenar o cargar el checkpoint
  4. evaluar en la suite fija
  5. registrar y verificar hashes
  6. criterio de selección
  7. publicar el puntero del modelo servido
  8. verificar la versión servida (si falla, rollback automático)

Cada ejecución deja un reporte en results/<nivel>/pipeline/<fecha>.json y una línea en
results/<nivel>/pipeline/decisiones.jsonl (promovido, rechazado o revertido, con el motivo).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("DO_NOT_TRACK", "1")                       # sin telemetría de Prefect
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

from prefect import flow, get_run_logger, task  # noqa: E402

from agents.registry import load_manifest, load_policy, sha256_file  # noqa: E402
from agents.serving import promote, rollback, served_run  # noqa: E402
from envs.levels import LEVEL_IDS, load_level, model_matches_level  # noqa: E402
from envs.parking_env import SUPPORTED_VERSIONS  # noqa: E402
from pipelines.scoring import evaluate, suite_for_level  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TIME_TOLERANCE = 0.02    # a igual éxito, el candidato debe ser al menos 2 % más rápido para reemplazar al servido


# ------------------------------------------------------------------ 1. validar
@task(name="1 · validar")
def validate(level_id: str, experiment: str | None, run_id: str | None, seed: int | None) -> dict:
    if level_id not in LEVEL_IDS:
        raise ValueError(f"Nivel desconocido: {level_id}")
    if bool(experiment) == bool(run_id):
        raise ValueError("Indique --experiment (entrenar) o --run-id (checkpoint existente), no ambos")
    version = load_level(level_id)["environment_version"]
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"{level_id} usa {version}, que el ambiente ya no soporta")
    suite = suite_for_level(level_id)
    if experiment:
        from pipelines.train import load_experiment, resolve
        config = load_experiment(experiment)
        if config["level_id"] != level_id:
            raise ValueError(f"{experiment} es de {config['level_id']}, no de {level_id}")
        if seed is not None:
            config["seed"] = seed
        resolve(config)                                   # valida presupuesto, currículo y suite
        return {"mode": "train", "config": config, "environment_version": version, "suite_id": suite["suite_id"]}
    manifest = load_manifest(run_id)
    if manifest["level_id"] != level_id or not model_matches_level(manifest, level_id):
        raise ValueError(f"{run_id} no es compatible con {level_id} ({manifest['environment_version']})")
    return {"mode": "load", "run_id": run_id, "environment_version": version, "suite_id": suite["suite_id"]}


# ------------------------------------------------------------------ 2. pruebas
@task(name="2 · pruebas")
def run_tests(quick: bool) -> str:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if quick:
        cmd += ["-k", "level or api or websocket or serving or rollback or evaluate or steps or garage"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-300:]
    if proc.returncode != 0:
        raise RuntimeError(f"Fallan las pruebas: {summary}")
    return summary


# ------------------------------------------------------------------ 3. entrenar o cargar
@task(name="3 · entrenar o cargar checkpoint")
def train_or_load(plan: dict) -> dict:
    if plan["mode"] == "load":
        return load_manifest(plan["run_id"])
    from pipelines.train import run_experiment
    return run_experiment(plan["config"])         # queda en MLflow y con manifiesto aunque falle


# ------------------------------------------------------------------ 4. evaluar
@task(name="4 · evaluar en la suite fija")
def evaluate_candidates(level_id: str, candidate: str, served: str | None, out_dir: Path) -> dict:
    """Candidato y modelo servido sobre los mismos casos de la suite (resultado por caso guardado)."""
    suite = suite_for_level(level_id)
    results = {}
    for role, run_id in (("candidate", candidate), ("served", served)):
        if run_id is None:
            continue
        result = evaluate(load_policy(load_manifest(run_id)), suite)
        (out_dir / f"eval_{role}.json").write_text(json.dumps({"run_id": run_id, **result}, indent=1))
        results[role] = {"run_id": run_id, **result["summary"]}
    return results


# ------------------------------------------------------------------ 5. registrar y verificar hashes
@task(name="5 · registrar y verificar hashes")
def verify_registration(manifest: dict) -> dict:
    if manifest["status"] != "completed":
        raise ValueError(f"{manifest['run_id']} no terminó: {manifest.get('diagnostic')}")
    model_sha = sha256_file(ROOT / manifest["model_uri"])
    if model_sha != manifest["model_sha256"]:
        raise ValueError(f"El hash del modelo de {manifest['run_id']} no coincide con su manifiesto")
    checks = {"model_sha256": model_sha}
    config_path = ROOT / manifest["config_uri"]
    if config_path.exists():
        from pipelines.train import config_hash
        if config_hash(json.loads(config_path.read_text())) != manifest["config_sha256"]:
            raise ValueError(f"La configuración de {manifest['run_id']} cambió después del run")
        checks["config_sha256"] = manifest["config_sha256"]
    mlflow_db = ROOT / "mlflow.db"
    if mlflow_db.exists():                        # el registro de MLflow vive en la máquina de entrenamiento
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        run = mlflow.get_run(manifest["mlflow"]["run_id"])
        if run.data.tags.get("run_id") != manifest["run_id"]:
            raise ValueError("El run de MLflow no corresponde al manifiesto")
        checks["mlflow_run"] = manifest["mlflow"]["run_id"]
    return checks


# ------------------------------------------------------------------ 6. criterio de selección
@task(name="6 · criterio de selección")
def select(evaluations: dict) -> dict:
    """Promueve si el candidato tiene más éxito en la suite, o el mismo éxito y es al menos 2 % más rápido."""
    cand = evaluations["candidate"]
    served = evaluations.get("served")
    if served is None:
        return {"promote": True, "reason": "El nivel no tenía modelo servido"}
    if cand["run_id"] == served["run_id"]:
        return {"promote": False, "reason": "El candidato ya es el modelo servido"}
    c_rate, s_rate = cand["success_rate"], served["success_rate"]
    if c_rate > s_rate:
        return {"promote": True, "reason": f"Más éxito en la suite: {c_rate:.0%} contra {s_rate:.0%}"}
    if c_rate == s_rate and cand["mean_time_s"] < served["mean_time_s"] * (1 - TIME_TOLERANCE):
        return {"promote": True, "reason": f"Mismo éxito ({c_rate:.0%}) y más rápido: "
                                           f"{cand['mean_time_s']} s contra {served['mean_time_s']} s"}
    return {"promote": False, "reason": f"No supera al servido: éxito {c_rate:.0%} contra {s_rate:.0%}, "
                                        f"tiempo {cand['mean_time_s']} s contra {served['mean_time_s']} s"}


# ------------------------------------------------------------------ 7. publicar
@task(name="7 · publicar modelo servido")
def publish(level_id: str, run_id: str, decision: dict, dry_run: bool) -> str | None:
    if not decision["promote"]:
        return None
    if dry_run:
        return f"(dry-run) se promovería {run_id}"
    promote(level_id, run_id, f"Pipeline: {decision['reason']}")
    return run_id


# ------------------------------------------------------------------ 8. verificar
@task(name="8 · verificar versión servida")
def verify_served(level_id: str, expected: str | None, promoted: bool) -> dict:
    """Una partida nueva del servidor debe usar el puntero y terminar episodios; si no, rollback."""
    from api.server import SimulationSession
    session = SimulationSession(level_id, public=True)
    problems = []
    if expected and session.agent_name != expected:
        problems.append(f"el servidor usa {session.agent_name}, se esperaba {expected}")
    for seed in range(3):
        session.reset(seed=seed)
        while session.status == "running":
            session.step()
    if problems and promoted:
        rollback(level_id, f"Pipeline: verificación fallida ({'; '.join(problems)})")
        return {"ok": False, "rolled_back_to": served_run(level_id), "problems": problems}
    if problems:
        raise RuntimeError("; ".join(problems))
    return {"ok": True, "served": session.agent_name}


@flow(name="parking-pipeline", log_prints=True)
def parking_pipeline(level: str, experiment: str | None = None, run_id: str | None = None,
                     seed: int | None = None, quick_tests: bool = True, dry_run: bool = False) -> dict:
    logger = get_run_logger()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = RESULTS_DIR / level / "pipeline" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = validate(level, experiment, run_id, seed)
    tests = run_tests(quick_tests)
    manifest = train_or_load(plan)
    served_before = served_run(level)
    evaluations = evaluate_candidates(level, manifest["run_id"], served_before, out_dir)
    registration = verify_registration(manifest)
    decision = select(evaluations)
    published = publish(level, manifest["run_id"], decision, dry_run)
    expected = manifest["run_id"] if published and not dry_run else served_before
    verification = verify_served(level, expected, bool(published) and not dry_run)

    outcome = ("revertido" if not verification["ok"] else "promovido" if published and not dry_run
               else "simulado" if published else "rechazado")
    report = {"level_id": level, "at": stamp, "mode": plan["mode"], "candidate": manifest["run_id"],
              "served_before": served_before, "served_after": served_run(level), "outcome": outcome,
              "decision": decision, "evaluations": evaluations, "registration": registration,
              "tests": tests, "verification": verification, "suite_id": plan["suite_id"],
              "environment_version": plan["environment_version"], "dry_run": dry_run}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with open(RESULTS_DIR / level / "pipeline" / "decisiones.jsonl", "a") as log:
        log.write(json.dumps({k: report[k] for k in ("at", "candidate", "served_before", "served_after",
                                                     "outcome", "decision", "dry_run")}, ensure_ascii=False) + "\n")
    logger.info(f"{outcome.upper()}: {decision['reason']} → servido: {report['served_after']}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", required=True)
    parser.add_argument("--experiment", help="configs/experiments/*.json (entrena un run nuevo)")
    parser.add_argument("--run-id", help="usa un checkpoint ya registrado")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--full-tests", action="store_true", help="corre todas las pruebas (por defecto, las rápidas)")
    parser.add_argument("--dry-run", action="store_true", help="decide sin mover el puntero del modelo servido")
    args = parser.parse_args(argv)
    report = parking_pipeline(args.level, args.experiment, args.run_id, args.seed,
                              quick_tests=not args.full_tests, dry_run=args.dry_run)
    print(json.dumps({k: report[k] for k in ("candidate", "served_before", "served_after", "outcome")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
