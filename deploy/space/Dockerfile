# Servidor de la simulación (API + WebSocket) para Hugging Face Spaces (Docker, puerto 7860).
# Solo sirve modelos ya entrenados: el entrenamiento se hace en local y queda en MLflow.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
    PARKING_PUBLIC=1 PARKING_LEVEL=N5

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=user . .
USER user

# PARKING_ALLOWED_ORIGINS (dominio de Vercel) y GIT_COMMIT se definen en la configuración del Space
EXPOSE 7860
CMD ["uvicorn", "api.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
