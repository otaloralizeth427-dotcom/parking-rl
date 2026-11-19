# Servidor de la simulación (API HTTPS + WebSocket) para Render.
# Contexto = raíz del repositorio; .dockerignore deja pasar solo el código, los niveles,
# los manifiestos y los archivos de modelo. Solo sirve modelos ya entrenados (se entrena en local).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
    PARKING_PUBLIC=1 PARKING_LEVEL=N5 PORT=10000

RUN useradd -m -u 1000 app
WORKDIR /home/app/srv

COPY deploy/requirements-serve.txt .
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=app . .
USER app

# Render fija PORT y RENDER_GIT_COMMIT; PARKING_ALLOWED_ORIGINS se define en el panel (dominio de Vercel)
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn api.server:create_app --factory --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
