"""Reconstruye el registro de MLflow de consulta a partir de la copia versionada.

    python deploy/build_registry.py <destino>

Descomprime registry/mlflow.db.gz en <destino>/mlflow.db, apunta las rutas de artefactos a
<destino>/mlartifacts y arma allí el paquete de cada run copiando artifacts/<nivel>/<run_id>/
(los mismos archivos que MLflow guardó: mlflow_snapshot.py lo comprobó con sha256).
"""

from __future__ import annotations

import gzip
import json
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARK = "file:///REGISTRY_ARTIFACTS"


def build(dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    db = dest / "mlflow.db"
    with gzip.open(ROOT / "registry" / "mlflow.db.gz", "rb") as f_in, open(db, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    target = (dest / "mlartifacts").resolve().as_uri()
    con = sqlite3.connect(db)
    for table, col in (("experiments", "artifact_location"), ("runs", "artifact_uri")):
        con.execute(f"UPDATE {table} SET {col} = ? || substr({col}, ?) WHERE {col} LIKE ?",
                    (target, len(MARK) + 1, MARK + "%"))
    con.commit()
    con.close()
    runs = json.loads((ROOT / "registry" / "runs.json").read_text())
    copied = 0
    for uuid, info in runs.items():
        package = dest / "mlartifacts" / uuid / "artifacts" / "package"
        package.mkdir(parents=True, exist_ok=True)
        for name in info["files"]:
            shutil.copy(ROOT / info["artifacts"] / name, package / name)
            copied += 1
    return {"db": str(db), "runs": len(runs), "files": copied}


if __name__ == "__main__":
    print(json.dumps(build(Path(sys.argv[1]))))
