"""Orquestación Prefect opcional para el pipeline del proyecto.

Instalación: ``pip install prefect>=3``.
Ejecución: ``python -m pipelines.prefect_flow configs/experiments/N4-exp04-ppo-ft-v6.json``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from prefect import flow, task

ROOT = Path(__file__).resolve().parents[1]


@task(retries=0)
def run_command(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    return result.stdout[-4000:]


@flow(name="parking-rl-experiment")
def experiment_flow(experiment: str, seed: int | None = None) -> str:
    run_command([sys.executable, "-m", "pytest", "-q"])
    command = [sys.executable, "run.py", "train", experiment]
    if seed is not None:
        command += ["--seed", str(seed)]
    return run_command(command)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    experiment_flow(args.experiment, args.seed)
