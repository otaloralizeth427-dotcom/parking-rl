# Despliegue y recuperación

## Desarrollo local

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
PARKING_ALLOWED_ORIGINS=http://127.0.0.1:8000 .venv/bin/python run.py serve --level N5
```

La interfaz y la API viven juntas en `http://127.0.0.1:8000`. Para una web estática
separada, publique `web/` en Vercel y configure el backend HTTPS en la variable
`PARKING_API_BASE_URL`; el backend debe permitir ese dominio mediante
`PARKING_ALLOWED_ORIGINS`.

## Pipeline reproducible

El flujo ligero deja estado por etapa en `runs/pipeline/`:

```bash
.venv/bin/python -m pipelines.flow configs/experiments/N4-exp04-ppo-ft-v6.json --seed 42
```

También existe una versión Prefect:

```bash
.venv/bin/python -m pip install 'prefect>=3'
.venv/bin/python -m pipelines.prefect_flow configs/experiments/N4-exp04-ppo-ft-v6.json --seed 42
```

## Catálogo, evaluación y recuperación

La API expone el catálogo compatible del nivel activo en `GET /api/catalog` y la
evaluación guardada de un modelo en `GET /api/model/<run_id>/evaluation`. Cambiar el
agente en la interfaz equivale a recuperar un modelo histórico compatible; antes de
cargarlo se valida el SHA-256 del manifiesto. El modelo recomendado se calcula por
mayor éxito en la suite fija, sin usar el test final reservado.

## MLflow

El entrenamiento usa `sqlite:///mlflow.db` y `mlartifacts/` localmente. Para la
revisión, levante MLflow en un servidor persistente y conserve la URL en la
documentación de la entrega; no suba credenciales ni la base local al repositorio.
