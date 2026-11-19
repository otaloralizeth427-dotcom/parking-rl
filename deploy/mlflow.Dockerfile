# Registro de experimentos (MLflow) de solo consulta, para el docente. Se construye desde git:
# registry/mlflow.db.gz (copia del registro, deploy/mlflow_snapshot.py) + artifacts/ (paquetes).
# Los cambios hechos desde la interfaz no persisten: cada despliegue vuelve a la copia versionada.
FROM python:3.12-slim
RUN pip install --no-cache-dir mlflow==3.18.0
RUN useradd -m -u 1000 app && mkdir -p /home/app/src && chown app:app /home/app/src
WORKDIR /home/app/src
COPY --chown=app registry/ registry/
COPY --chown=app artifacts/ artifacts/
COPY --chown=app deploy/build_registry.py deploy/
USER app
RUN python deploy/build_registry.py /home/app/registry && rm -rf artifacts

# Se lanza la app de MLflow directamente (sin el proceso lanzador ni los trabajadores de tareas
# en segundo plano), para caber en los 512 MB del plan gratuito: ~250 MB medidos.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=10000 \
    MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_ENABLE_WORKSPACES=false \
    MLFLOW_TRACKING_URI=sqlite:////home/app/registry/mlflow.db \
    MLFLOW_REGISTRY_URI=sqlite:////home/app/registry/mlflow.db \
    _MLFLOW_SERVER_FILE_STORE=sqlite:////home/app/registry/mlflow.db \
    _MLFLOW_SERVER_REGISTRY_STORE=sqlite:////home/app/registry/mlflow.db \
    _MLFLOW_SERVER_ARTIFACT_ROOT=mlflow-artifacts:/ \
    _MLFLOW_SERVER_SERVE_ARTIFACTS=true \
    _MLFLOW_SERVER_ARTIFACT_DESTINATION=/home/app/registry/mlartifacts \
    MLFLOW_SERVER_X_FRAME_OPTIONS=SAMEORIGIN _MLFLOW_SGI_NAME=uvicorn \
    MLFLOW_SERVER_ALLOWED_HOSTS=parking-rl-mlflow.onrender.com,localhost,127.0.0.1
EXPOSE 10000
CMD ["sh", "-c", "exec python -m uvicorn mlflow.server.fastapi_app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
