"""Copia de consulta del registro de MLflow, versionada en git y publicable.

    python deploy/mlflow_snapshot.py           # → registry/mlflow.db.gz + registry/runs.json

El registro original (mlflow.db + mlartifacts/) vive en la máquina de entrenamiento. Esta copia:
  - reescribe las rutas de artefactos para que no dependan de carpetas personales;
  - no duplica artefactos: los paquetes de cada run son idénticos a artifacts/<nivel>/<run_id>/
    (se comprueba archivo por archivo con sha256) y se reconstruyen al construir la imagen
    (deploy/build_registry.py).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "registry"
SERVED_ARTIFACTS = "file:///REGISTRY_ARTIFACTS"     # marcador: build_registry.py pone la ruta real


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    src_db = ROOT / "mlflow.db"
    old_root = (ROOT / "mlartifacts").as_uri()
    runs = {}
    for path in sorted((ROOT / "manifests").glob("*.json")):
        m = json.loads(path.read_text())
        uuid = m["mlflow"]["run_id"]
        package = ROOT / "mlartifacts" / uuid / "artifacts" / "package"
        source = ROOT / "artifacts" / m["level_id"] / m["run_id"]
        if package.is_dir():
            for f in package.iterdir():
                if not (source / f.name).exists() or sha(f) != sha(source / f.name):
                    sys.exit(f"{m['run_id']}: {f.name} difiere entre mlartifacts y artifacts; no se publica")
        runs[uuid] = {"run_id": m["run_id"], "artifacts": str(source.relative_to(ROOT)),
                      "files": sorted(f.name for f in package.iterdir()) if package.is_dir() else []}
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "mlflow.db"
        shutil.copy(src_db, db)
        con = sqlite3.connect(db)
        con.execute("UPDATE experiments SET artifact_location = ? || substr(artifact_location, ?) "
                    "WHERE artifact_location LIKE ?", (SERVED_ARTIFACTS, len(old_root) + 1, old_root + "%"))
        con.execute("UPDATE experiments SET artifact_location = ? WHERE experiment_id = 0",
                    (SERVED_ARTIFACTS + "/0",))
        con.execute("UPDATE runs SET artifact_uri = ? || substr(artifact_uri, ?) WHERE artifact_uri LIKE ?",
                    (SERVED_ARTIFACTS, len(old_root) + 1, old_root + "%"))
        left = con.execute("SELECT count(*) FROM runs WHERE artifact_uri LIKE '%/Users/%'").fetchone()[0]
        if left:
            sys.exit(f"Quedaron {left} rutas personales en la copia")
        con.commit()
        con.execute("VACUUM")
        con.close()
        OUT.mkdir(exist_ok=True)
        with open(db, "rb") as f_in, gzip.open(OUT / "mlflow.db.gz", "wb", compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
    (OUT / "runs.json").write_text(json.dumps(runs, indent=1) + "\n")
    print(f"registry/mlflow.db.gz ({(OUT / 'mlflow.db.gz').stat().st_size / 1e6:.1f} MB) · {len(runs)} runs con paquete")


if __name__ == "__main__":
    main()
