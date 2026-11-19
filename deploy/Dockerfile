# Servidor de la simulación (API HTTPS + WebSocket) para Render.
# Contexto = raíz del repositorio. Solo sirve modelos ya entrenados: se entrena en local.

# Etapa 1: de artifacts/ quedan solo los archivos de modelo (curvas y evaluaciones no hacen falta)
FROM python:3.12-slim AS models
WORKDIR /src
COPY artifacts/ artifacts/
RUN find artifacts -type f ! -name 'model.zip' ! -name 'model.npz' -delete

# Etapa 2: el servidor
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
    PARKING_PUBLIC=1 PARKING_LEVEL=N5 PORT=10000

RUN useradd -m -u 1000 app && mkdir -p /home/app/srv && chown app:app /home/app/srv
WORKDIR /home/app/srv

COPY deploy/requirements-serve.txt .
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=app api/ api/
COPY --chown=app agents/ agents/
COPY --chown=app envs/ envs/
COPY --chown=app pipelines/__init__.py pipelines/scoring.py pipelines/
COPY --chown=app configs/levels/ configs/levels/
COPY --chown=app configs/suites/ configs/suites/
COPY --chown=app manifests/ manifests/
COPY --chown=app serving/ serving/
COPY --chown=app --from=models /src/artifacts/ artifacts/
USER app

# Render fija PORT y RENDER_GIT_COMMIT; PARKING_ALLOWED_ORIGINS se define en el panel (dominio de Vercel)
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn api.server:create_app --factory --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
