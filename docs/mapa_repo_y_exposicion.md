# Mapa del repositorio y orden de la exposición

## 1. Estructura del repositorio

| Carpeta / archivo | Qué hay adentro | Para qué sirve (y qué pide el PDF) |
|---|---|---|
| `envs/` | El mundo del juego: parqueadero, auto, sensores, peatones, garaje grande (`garage.py`) y freno de emergencia (`safety.py`) | El ambiente (Gymnasium) con sus versiones `parking-v2` … `v14`; descrito en `docs/ficha_ambiente.md` |
| `configs/levels/` | Un archivo por nivel (N1…N6): versión del ambiente y qué contiene | Niveles versionados (mínimo 3) |
| `configs/experiments/` | 35 recetas de entrenamiento, cada una con su hipótesis y su cambio frente a la anterior | 4 o más configuraciones con hipótesis, incluidas las fallidas |
| `configs/suites/` | Pruebas fijas de 100 casos y huellas de los exámenes finales secretos | Evaluación reproducible y test reservado |
| `agents/` | PPO, DQN, Q-learning, regla fija; `registry.py` (carga con verificación del hash) y `serving.py` (modelo en uso y rollback) | Algoritmo, referencias, identidad y rollback |
| `pipelines/` | `train.py`, `evaluate.py`, `flow.py` (pipeline de 8 pasos con Prefect) | Pipeline completo |
| `artifacts/` | Paquete de cada entrenamiento: modelo, receta, curvas y evaluación por caso | Modelos recuperables |
| `manifests/` | Cédula de cada modelo: commit, versión del ambiente, semilla y hashes SHA-256 | Manifiesto por modelo |
| `serving/` | Modelo en uso por nivel, con historial de promociones y rollbacks | Modelo servido y prueba de rollback |
| `results/` | Comparaciones, curvas, exámenes finales y decisiones del pipeline | Evaluaciones por caso y comparación |
| `registry/` | Copia publicable del registro de MLflow | Acceso al registro de experimentos |
| `api/` | Servidor FastAPI: una partida por visitante, evaluación, comparación y monitoreo | API HTTPS, CORS y monitoreo |
| `web/` | Juego 3D (TypeScript + Vite) y presentación (`public/presentacion.html`) | Página en Vercel |
| `deploy/`, `render.yaml` | Docker del servidor y de MLflow para Render | Despliegue |
| `tests/` | 83 pruebas automáticas (corren en GitHub Actions) | Calidad |
| `docs/` | Ficha, experimentos, pipeline, rollback y monitoreo, despliegue e informe | Documentación de la entrega |
| `README.md` | Misión, arquitectura, instalación, ejecución y reproducción | Punto de entrada |

## 2. Orden de la exposición (10 minutos)

| # | Tema | Qué explicar | Qué mostrar | Dónde está |
|---|---|---|---|---|
| 1 | El problema | Estacionarse solo, sin chocar ni salirse; ahorra tiempo, evita roces y protege peatones | Diapositiva 2 | `README.md` |
| 2 | El ambiente | Qué ve, 5 acciones, +100 / −50, terminated o truncated | Diapositiva 3 | `envs/`, `docs/ficha_ambiente.md` |
| 3 | Los niveles | 6 niveles; cada cambio del mundo sube la versión (v2 → v14) | Diapositiva 4 | `configs/levels/` |
| 4 | El modelo | PPO con currículo, continuar modelos, 3 semillas | Diapositiva 5 | `agents/`, `configs/experiments/` |
| 5 | Resultados | Aleatorio 0 %, regla 9–49 %, IA 95–100 %; prueba fija + examen secreto | Diapositiva 6 | `results/`, `configs/suites/` |
| 6 | Lo que no funcionó | Conos invisibles, conos que tapaban el giro, «usted primero», atropellos, pasillos angostos, N6 en curso | Diapositiva 7 | `docs/experimentos.md` |
| 7 | Versionamiento | Receta → entrenamiento → MLflow → manifiesto → evaluación → modelo servido; ramas y tags | Diapositiva 8 + MLflow | `manifests/`, `serving/` |
| 8 | Pipeline, despliegue y monitoreo | 8 pasos en Prefect; Vercel + Render; monitoreo | Diapositiva 9 | `pipelines/flow.py`, `docs/pipeline.md` |
| 9 | Demo | N5 → N3 (frena por la peatona) → Métricas → Comparar → MLflow → rollback | Juego publicado | presentación (`web/public/presentacion.html`) |
| 10 | Limitaciones y aprendizajes | Física simplificada, freno con posición real del peatón, servidor gratuito | Diapositiva 11 | `docs/informe.md` |
| 11 | Cierre | Enlaces y preguntas | Diapositiva 12 | — |
