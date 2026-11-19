# Presentación final · AutoPark RL

Guion de 10 minutos para la sustentación. La demostración se hace con la versión
publicada; no es necesario entrenar durante la presentación.

## Enlaces y versión

- Web: <https://parking-rl.vercel.app/?level=N5>
- API: <https://parking-rl-api.onrender.com/api/health>
- MLflow: <https://parking-rl-mlflow.onrender.com>
- Repositorio: <https://github.com/otaloralizeth427-dotcom/parking-rl>
- Release: `v1.0`
- Commit: `dd55d5780f69519e31e7fa81e2f558788fcc06cf`

## Diapositivas y tiempo

### 1. Problema y objetivo · 45 segundos

Un vehículo debe llegar a un cupo libre y estacionarse con orientación correcta,
evitando autos, columnas, conos y peatones. El valor está en reducir maniobras,
tiempo y riesgos en un parqueadero real.

### 2. Ambiente y contrato · 1 minuto

- Ambiente Gymnasium `ParkingEnv`.
- Acciones discretas: avanzar, retroceder, izquierda, derecha y frenar.
- Paso físico de `0.2 s`.
- Observaciones numéricas con posición relativa del cupo, orientación, velocidad y sensores.
- `terminated`: éxito o colisión.
- `truncated`: agotamiento del presupuesto de pasos.
- La recompensa premia progreso y estacionamiento, y penaliza pasos, choques y salidas.

### 3. Evolución de la dificultad · 1 minuto

| Nivel | Nueva decisión |
|---|---|
| N1 | Estacionar desde el pasillo |
| N2 | Rodear conos |
| N3 | Ceder el paso y retomar ante peatones |
| N4 | Combinar obstáculos, peatones y ruido |
| N5 | Buscar una ruta larga en el garaje |

Los niveles y las versiones del ambiente están en `configs/levels/`.

### 4. Algoritmo y entrenamiento · 1 minuto

Usamos PPO como algoritmo principal, con DQN y Q-learning como comparaciones.
Cada experimento tiene hipótesis, presupuesto, semilla, configuración y modelo padre.
El entrenamiento se registra en MLflow y la evaluación usa el modelo congelado y una
suite fija de 100 episodios.

### 5. Resultados · 1 minuto 30 segundos

| Nivel | Modelo servido | Éxito en suite |
|---|---|---:|
| N1 | `N1-exp03-ppo-s42-64bf7433` | 97 % |
| N2 | `N2-exp06-ppo-curriculo-3m-s44-7f36a120` | 80 % |
| N3 | `N3-exp04-ppo-ft-v6-s43-bd380c76` | 93 % |
| N4 | `N4-exp04-ppo-ft-v6-s43-e098d962` | 85 % |
| N5 | `N5-exp05-ppo-inverso-3m-s44-78c82696` | 100 % |

También mostramos candidatos rechazados: DQN de N1, PPO sin currículo y currículos
que dependían demasiado de la semilla. No ocultamos los intentos que empeoraron.

### 6. Trazabilidad MLOps · 1 minuto

Cada modelo tiene manifiesto con run, commit, ambiente, configuración, semilla,
normalización, hash SHA-256 y evaluación. El catálogo web permite seleccionar un
modelo histórico, consultar sus métricas y ejecutar una comparación.

### 7. Demostración de Vercel · 2 minutos

1. Abrir <https://parking-rl.vercel.app/?level=N5>.
2. Abrir el panel y cambiar entre N1, N3 y N5.
3. Mostrar el modelo servido y sus métricas.
4. Seleccionar dos modelos y pulsar **Comparar**.
5. Activar un modelo histórico y mostrar que la partida cambia de agente.
6. Ejecutar un episodio con el botón de reproducción.
7. Abrir la API `/api/health` y mostrar commit, niveles y modelo servido.
8. Abrir MLflow y mostrar un run con parámetros, métricas y artefactos.

### 8. Incidente, intervención y rollback · 1 minuto

En N3 el auto y el peatón podían quedarse esperando mutuamente. Se midieron 4
atascos en 100 episodios, se cambió la dinámica a una versión nueva y se reentrenó.
En N5 se promovió un candidato, se comparó con el modelo servido y se hizo rollback
porque empataba en éxito pero era más lento. El historial queda versionado en `serving/`.

### 9. Limitaciones y cierre · 45 segundos

- Render puede tardar en despertar después de inactividad.
- El monitoreo operativo se reinicia si el servicio se reinicia.
- N2 y N4 todavía tienen fallos y los atropellos no serían aceptables en un vehículo real.
- La física y los peatones son simulaciones simplificadas.

Conclusión: el aporte principal no es solo el porcentaje de éxito; es poder explicar,
recuperar y comparar cada resultado desde el ambiente hasta el modelo servido.

## Repartición sugerida

- **Juan Esteban Molina:** ambiente, pipeline, métricas, manifiestos y rollback.
- **Nathalia Otalora:** niveles, experimentos, interfaz, Vercel y demostración.

Ambos deben poder explicar el contrato del ambiente, el versionamiento y el flujo de
recuperación.
