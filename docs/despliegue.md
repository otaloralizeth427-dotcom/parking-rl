# Despliegue y recuperación

## Arquitectura publicada

| Parte | Dónde | Cómo se publica |
|---|---|---|
| Web 3D (`web/`, TypeScript + Vite) | Vercel: https://parking-rl.vercel.app | Proyecto `parking-rl`. Root Directory `web`, `npm ci`, `npm run build`, salida `dist`. Variables `VITE_API_BASE_URL` y `VITE_MLFLOW_URL`. Cada push a `main` vuelve a publicar. |
| API de la simulación | Render, servicio `parking-rl-api` | `render.yaml` + `deploy/Dockerfile`. HTTPS y WebSocket, una partida por visitante. |
| Registro MLflow (consulta) | Render, servicio `parking-rl-mlflow` | `deploy/mlflow.Dockerfile`, armado desde `registry/` y `artifacts/`. |

**Variables:**
- del servidor: `.env.example`;
- de la web: `web/.env.example`.

Ninguna contiene secretos. El servidor solo acepta los orígenes de la web publicada.

## Desarrollo local

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock.txt
cd web && npm ci && npm run build && cd ..
.venv/bin/python run.py serve --level N5          # web y API juntas en http://127.0.0.1:8000
```

`cd web && npm run dev` abre la web con recarga en caliente. Usa como API el servidor local del puerto 8765
(se cambia con `PARKING_API`).

## Pipeline

- **Completo, con Prefect** ([pipeline.md](pipeline.md)): `python -m pipelines.flow --level N5 --run-id <run>`.
  Hace los 8 pasos y termina con publicación, verificación y rollback.
- **Ligero, sin Prefect**: deja el estado de cada etapa en `runs/pipeline/`.

  ```bash
  .venv/bin/python -m pipelines.flow_simple configs/experiments/N4-exp04-ppo-ft-v6.json --seed 42
  .venv/bin/python -m pipelines.prefect_flow configs/experiments/N4-exp04-ppo-ft-v6.json --seed 42
  ```

## Catálogo, evaluación y recuperación

- `GET /api/catalog?level=N5`: modelos del nivel, con hipótesis, presupuesto, suite y métricas por causa.
- `GET /api/model/<run_id>/evaluation`: la evaluación guardada de un modelo, caso por caso.
- `GET /api/compare?run_ids=a,b` y `POST /api/rollback` `{ "run_id": … }`: comparan métricas guardadas y
  activan un modelo histórico en la partida REST.
- **En la web**, la ficha del modelo permite activar esa versión en la partida del visitante. «Evaluar y
  comparar» corre varios modelos en vivo sobre los mismos casos de la suite.
- **El modelo servido a todos** es el puntero versionado `serving/<nivel>.json`. Se cambia con
  `python -m agents.serving promote|rollback`; ver [rollback y monitoreo](rollback_y_monitoreo.md).
- Antes de cargar cualquier modelo se verifica el SHA-256 de su manifiesto. Activar otra versión no
  sobrescribe ni elimina la anterior.

## MLflow

- **Entrenamiento:** usa `sqlite:///mlflow.db` y `mlartifacts/` en la máquina del equipo. Esos archivos no se
  suben.
- **Copia publicable:** `python deploy/mlflow_snapshot.py` arma `registry/`, sin rutas personales y sin
  duplicar artefactos. Render la publica como MLflow de solo consulta.
