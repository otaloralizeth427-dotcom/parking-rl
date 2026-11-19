"""Modelo servido por nivel: puntero versionado en git, con historial y rollback.

serving/<NIVEL>.json guarda qué run se sirve («current») y un historial que solo crece: cada
promoción o rollback agrega una entrada, así siempre queda rastro de qué se sirvió, cuándo y por qué.

    python -m agents.serving status  N5
    python -m agents.serving promote N5 <run_id> --reason "gana en la suite"
    python -m agents.serving rollback N5 --reason "degradación en producción"

Antes de mover el puntero se comprueba que el run esté completo, sea compatible con la versión del
ambiente del nivel y que el hash del modelo coincida con su manifiesto (se carga de verdad).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agents.registry import load_manifest, load_policy

ROOT = Path(__file__).resolve().parents[1]
SERVING_DIR = ROOT / "serving"


def _path(level_id: str) -> Path:
    return SERVING_DIR / f"{level_id}.json"


def load_serving(level_id: str) -> dict | None:
    path = _path(level_id)
    return json.loads(path.read_text()) if path.exists() else None


def served_run(level_id: str) -> str | None:
    serving = load_serving(level_id)
    return serving["current"]["run_id"] if serving else None


def _check(level_id: str, run_id: str) -> dict:
    from envs.levels import model_matches_level

    manifest = load_manifest(run_id)
    if manifest["level_id"] != level_id:
        raise ValueError(f"{run_id} es de {manifest['level_id']}, no de {level_id}")
    if manifest["status"] != "completed":
        raise ValueError(f"{run_id} no está completo ({manifest['status']})")
    if not model_matches_level(manifest, level_id):
        raise ValueError(f"{run_id} se entrenó con otra versión del nivel ({manifest['environment_version']})")
    load_policy(manifest)                       # verifica el hash y que el modelo realmente cargue
    return manifest


def _write(level_id: str, run_id: str, action: str, reason: str) -> dict:
    manifest = _check(level_id, run_id)
    serving = load_serving(level_id) or {"level_id": level_id, "history": []}
    entry = {
        "run_id": run_id, "action": action, "reason": reason,
        "environment_version": manifest["environment_version"], "model_sha256": manifest["model_sha256"],
        "test_success_rate": (manifest.get("evaluation_summary") or {}).get("success_rate"),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    serving["current"] = entry
    serving["history"].append(entry)
    SERVING_DIR.mkdir(exist_ok=True)
    _path(level_id).write_text(json.dumps(serving, indent=2, ensure_ascii=False) + "\n")
    return serving


def promote(level_id: str, run_id: str, reason: str) -> dict:
    current = served_run(level_id)
    if current == run_id:
        raise ValueError(f"{run_id} ya es el modelo servido de {level_id}")
    return _write(level_id, run_id, "promote", reason)


def previous_run(level_id: str) -> str | None:
    """El modelo que se servía justo antes del actual (saltando repeticiones)."""
    serving = load_serving(level_id)
    if not serving:
        return None
    current = serving["current"]["run_id"]
    for entry in reversed(serving["history"]):
        if entry["run_id"] != current:
            return entry["run_id"]
    return None


def rollback(level_id: str, reason: str) -> dict:
    target = previous_run(level_id)
    if target is None:
        raise ValueError(f"{level_id} no tiene una versión anterior a la cual volver")
    return _write(level_id, target, "rollback", reason)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Modelo servido por nivel (promover / rollback)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").add_argument("level")
    p = sub.add_parser("promote")
    p.add_argument("level")
    p.add_argument("run_id")
    p.add_argument("--reason", required=True)
    r = sub.add_parser("rollback")
    r.add_argument("level")
    r.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "status":
        serving = load_serving(args.level)
        print(json.dumps(serving, indent=2, ensure_ascii=False) if serving else f"{args.level} sin puntero")
    elif args.cmd == "promote":
        print(json.dumps(promote(args.level, args.run_id, args.reason)["current"], indent=2, ensure_ascii=False))
    else:
        print(json.dumps(rollback(args.level, args.reason)["current"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
