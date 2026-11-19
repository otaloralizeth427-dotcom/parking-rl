"""Pipeline ligero (sin Prefect): pruebas → entrenar, evaluar y registrar, con estado por etapa.

Uso:
    python -m pipelines.flow_simple configs/experiments/N4-exp04-ppo-ft-v6.json --seed 42

El pipeline completo de 8 pasos (con selección, publicación y rollback) es pipelines/flow.py.

Es una orquestación ligera sin estado externo: cada etapa deja un estado JSON en
``runs/pipeline/<run>`` y delega el trabajo de entrenamiento al pipeline existente.
Así puede ejecutarse en la máquina del equipo o dentro de un job de CI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.experiment.read_text())
    run_name = f"{config['experiment_id']}-s{args.seed or config['seed']}"
    out = ROOT / "runs" / "pipeline" / run_name
    out.mkdir(parents=True, exist_ok=True)
    state = {"run": run_name, "started_at": datetime.now(timezone.utc).isoformat(), "stages": []}

    def stage(name: str, command: list[str]) -> None:
        record = {"name": name, "command": command, "status": "running"}
        state["stages"].append(record)
        (out / "state.json").write_text(json.dumps(state, indent=2))
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        record.update({"status": "completed" if completed.returncode == 0 else "failed",
                       "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]})
        (out / "state.json").write_text(json.dumps(state, indent=2))
        if completed.returncode:
            raise SystemExit(completed.returncode)

    if not args.skip_tests:
        stage("tests", [sys.executable, "-m", "pytest", "-q"])
    stage("train-evaluate-register", [sys.executable, "run.py", "train", str(args.experiment),
                                       *( [] if args.seed is None else ["--seed", str(args.seed)] )])
    state["status"] = "completed"
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out / "state.json").write_text(json.dumps(state, indent=2))
    print(out / "state.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
