"""Arma la carpeta que se sube al Space de Hugging Face (servidor de la simulación).

Copia solo lo necesario para servir: el código del ambiente, los agentes y la API, los niveles,
los manifiestos y los archivos de modelo que esos manifiestos referencian (con su hash verificado).

    python deploy/build_space.py build/space
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.registry import sha256_file  # noqa: E402

CODE = ["api", "agents", "envs", "configs/levels"]


def build(out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("Dockerfile", "README.md", "requirements-serve.txt"):
        shutil.copy(ROOT / "deploy" / "space" / name, out / name)
    for rel in CODE:
        shutil.copytree(ROOT / rel, out / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    models = 0
    for path in sorted((ROOT / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text())
        (out / "manifests").mkdir(exist_ok=True)
        shutil.copy(path, out / "manifests" / path.name)
        uri = manifest.get("model_uri")
        if manifest.get("status") != "completed" or not uri:
            continue
        src = ROOT / uri
        if sha256_file(src) != manifest["model_sha256"]:
            raise SystemExit(f"El hash de {uri} no coincide con su manifiesto; no se publica")
        (out / uri).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out / uri)
        models += 1
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "SERVED_COMMIT").write_text(commit + "\n")
    return {"out": str(out), "models": models, "commit": commit}


if __name__ == "__main__":
    print(json.dumps(build(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build" / "space"))))
