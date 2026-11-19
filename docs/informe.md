# Informe final · AutoPark RL (escenario Autonomous Parking)

MLOps · Visualización 2 · Universidad Externado de Colombia.

Este informe resume las decisiones, los resultados, los límites y las contribuciones del proyecto. El detalle
está en documentos aparte:
- contrato del ambiente: [ficha del ambiente](ficha_ambiente.md);
- cada experimento con su hipótesis: [registro de experimentos](experimentos.md);
- operación: [pipeline](pipeline.md) y [rollback y monitoreo](rollback_y_monitoreo.md).

## 1. Misión y valor

Un vehículo debe llegar a un cupo libre y detenerse dentro de él, bien orientado, sin chocar autos,
columnas, conos ni peatones, y sin salirse del área. En un parqueadero real, que el auto estacione solo
ahorra tiempo, evita roces y aprovecha mejor el espacio. Por eso cada nivel agrega lo que hace difícil la
tarea en la vida real:
- obstáculos que hay que rodear;
- peatones que cruzan;
- sensores imprecisos;
- un garaje grande donde hay que buscar la ruta.

## 2. Ambiente (resumen)

- **Ambiente:** Gymnasium propio, `ParkingEnv`, con modelo cinemático de bicicleta y `dt = 0.2 s`.
- **Acciones:** `Discrete(5)`: avanzar, retroceder, izquierda, derecha y frenar.
- **Observación:** posición del cupo en el marco del auto, error de orientación, velocidad, dirección,
  indicador "dentro del cupo" y distancias de 8 sensores radiales. Desde `parking-v3` se suman 5 sensores de
  parachoques, y en el garaje, la pose en el mapa.
- **Recompensa:** progreso hacia el cupo (por ruta en el garaje), bono por estacionar y castigos por choque,
  por salir del área y por cada paso.
- **Fin del episodio:** `terminated` si estaciona o choca; `truncated` si se acaba el presupuesto de pasos.

| Nivel | Versión del ambiente | Qué agrega |
|---|---|---|
| N1 · Misión base | `parking-v2` | Pasillo con dos filas de autos y columnas |
| N2 · Conos en el pasillo | `parking-v13` | Conos, uno de ellos bloqueando la ruta; 5 sensores de parachoques. Los conos nunca tapan el giro hacia el cupo, el bloqueante se rodea por ambos lados y el pasillo mide 11 m |
| N3 · Peatones cruzando | `parking-v11` | Peatones que cruzan y siempre se mueven; desde v11, freno de emergencia y zona de precaución |
| N4 · Todo junto | `parking-v11` | Conos + peatones + ruido en los sensores (σ = 0.1), con el freno de emergencia de v11 |
| N5 · Garaje grande | `parking-v8` | Azotea de 40 × 50 m con 3 filas; hay que rodear el parqueadero para llegar al cupo |

## 3. Decisiones clave

1. **PPO de Stable-Baselines3** como algoritmo principal, que es la referencia que pide el curso. DQN y
   Q-learning tabular quedaron como comparación: no superaron a la regla en N1.
2. **Currículo por etapas** desde N2: se empieza cerca del cupo y se aleja. Sin él, PPO dependía de encontrar
   un éxito por suerte (N1 exp03: 97 / 0 / 0 % según la semilla).
3. **Continuar desde un checkpoint** (`parent_run_id`) en N3 y N4: mismo resultado con un tercio de los pasos.
   Además, importó **desde qué modelo** se parte: el de conos aprende N4 mejor que el de peatones.
4. **Currículo inverso por la ruta en N5:** se empieza junto al cupo y se aleja por el camino de entrada.
   Tamaño de red y γ no cambiaron el resultado.
5. **Cambiar el ambiente cuando el fallo es del ambiente, no del agente.** Cada cambio sube la versión y se
   prueba:
   - **conos invisibles** entre los rayos radiales: sensores de parachoques (v3);
   - **peatón y auto esperándose mutuamente:** el peatón cede (v4) y nunca se queda quieto (v6);
   - **pasillos tan angostos que la maniobra final era casi imposible:** pasillos de 8.5 m (v8).
6. **Elegir con la suite y confirmar con el test final reservado.** Las semillas del test final no están en
   el repositorio; solo su huella SHA-256. Se crea uno nuevo por cada versión del ambiente.
7. **Seguridad ante peatones como parte del ambiente (v10–v11).** Un freno de emergencia y una zona de
   precaución. El aprendizaje por refuerzo bajó los atropellos, pero no los llevaba a cero; con el freno, 0 en
   los tests finales.
8. **Puntero versionado al modelo servido** (`serving/`), con historial, rollback y un pipeline que solo
   promueve si el candidato gana en la suite.

## 4. Resultados

En todos los niveles se usaron 100 episodios con modelo congelado. La política aleatoria tuvo **0 %** en
todos los niveles.

| Nivel | Regla (suite) | **Modelo servido** (suite) | Test final reservado (regla / modelo) | Modelo servido |
|---|---|---|---|---|
| N1 | 49 % | **97 %** · 6.6 s | 53 % / **97 %** (`N1-final-v1`) | `N1-exp03-ppo-s42-64bf7433` |
| N2 | 19 % | **96 %** · 6.8 s | 9 % / **92 %** (`N2-final-v4`) | `N2-exp10-ppo-ft-v13-s42-c7bdedfb` |
| N3 | 43 % | **96 %** · 10.0 s, 0 atropellos | 43 % / **97 %** (`N3-final-v5`) | `N3-exp06-ppo-ft-v11-s42-2a9f8d11` |
| N4 | 16 % | **91 %** · 10.6 s, 0 atropellos | 11 % / **91 %** (`N4-final-v4`) | `N4-exp06-ppo-ft-v11-s42-4a905c49` |
| N5 | 30 % | **100 %** · 14.4 s | 21 % / **100 %** (`N5-final-v1`) | `N5-exp05-ppo-inverso-3m-s44-78c82696` |

En total hay **24 configuraciones** de experimento, cada una con hipótesis, y **56 runs completos** en MLflow
con su manifiesto. Además se conserva **1 run interrumpido** (KILLED), con su diagnóstico.

**Intentos que no mejoraron:**

| Nivel | Intento | Resultado |
|---|---|---|
| N1 | DQN (exp01, exp04) | 42 / 0 / 8 % y 0 % |
| N1 | PPO sin currículo (exp02) | 0 % |
| N2 | PPO desde cero con 8 ambientes en paralelo (exp03, exp04) | 0 % en las 6 semillas |
| N4 | Currículo desde cero | Dejó de ser confiable: 76 % con una semilla y 38 % con otra |
| N5 | 4 pilotos con pasillos angostos | 10, 0, 7 y 12 %. Diagnóstico: la maniobra final casi no cabía |

## 5. Versionamiento y trazabilidad

- **Código:**
  - ramas `feature/nivel-N` y `fix/...`, unidas a `main` con `--no-ff`;
  - un tag por nivel (`nivel-1-v1.0` … `nivel-5-v1.0`) y un tag de entrega `v1.0`.
- **Ambiente:** `ENV_VERSION` (`parking-v2` … `parking-v14`). Cada nivel fija su versión en
  `configs/levels/NX.json`, y un modelo solo se ofrece en el nivel si coincide (`model_matches_level`).
- **Configuraciones:**
  - un JSON por experimento, con `hypothesis` y `change_vs_baseline`;
  - el pipeline guarda la configuración resuelta e inmutable y su `config_sha256`.
- **Modelos:**
  - un manifiesto por run con `run_id`, `level_id`, `environment_version`, `git_commit`, `config_sha256`,
    `model_uri`, `model_sha256`, `normalization_uri`, `runtime_ref`, `evaluation_suite_id`, `seed` y
    `parent_run_id`;
  - el hash se verifica cada vez que se carga un modelo.
- **Registro:**
  - MLflow guarda parámetros, métricas por paso, etiquetas (commit, hipótesis, hash de configuración) y el
    paquete de cada run;
  - hay una copia versionada en `registry/`, publicada como servicio de solo consulta.

## 6. Operación

- **Pipeline (Prefect).** 8 pasos: validar, pruebas, entrenar o cargar, evaluar, verificar hashes,
  seleccionar, publicar y verificar con rollback automático. Hay dos ejecuciones reales registradas, ambas
  rechazadas, con su motivo.
- **Incidente de degradación (N3).** El auto y el peatón se esperaban mutuamente.
  - Medición: 4 de 100 episodios terminaban en atasco.
  - Intervención: el peatón cede (v4).
  - Resultado: 0 atascos.
  - Después se exigió que nunca se quede quieto (v6) y se reentrenó, con un test final nuevo.
- **Rollback.** Simulacro real en N5: se promovió un candidato, se comparó en el servidor y se volvió al
  anterior. `/api/health` confirmó cada paso.
- **Monitoreo.** `/api/metrics` y un panel en la web muestran:
  - éxito y causas de fallo por nivel y agente;
  - latencia p95 por paso;
  - partidas activas, commit y modelo servido.
- **Publicación.**
  - web en Vercel (TypeScript + Vite);
  - API en Render (Docker, HTTPS, una partida por visitante);
  - MLflow de consulta en Render;
  - integración continua en GitHub Actions.

## 7. Limitaciones conocidas

- **Plan gratuito de Render:**
  - el servidor se duerme tras 15 min sin uso, y la primera visita puede tardar hasta un minuto;
  - las métricas de monitoreo viven en memoria y se reinician con el servidor.
- **MLflow publicado:** es de solo consulta. Lo que se cambie desde su interfaz no persiste. La fuente de
  verdad es el registro de la máquina de entrenamiento y su copia en git.
- **Promover o revertir el modelo servido** se hace con commit y push del puntero, que Render vuelve a
  publicar. Desde la web el rollback aplica a la partida del visitante.
- **N2 y N4 todavía fallan:**
  - N2: 4–9 % de los episodios (tiempo agotado y algunos roces con conos);
  - N4: 9 % (tiempo agotado y conos).
  - Los atropellos de N3 y N4 bajaron a 0 con el freno de emergencia (v11), pero el freno usa la posición real
    del peatón; un auto real dependería de sensores con errores.
- **Simplificaciones de la física:** acciones discretas, cinemática sin derrape ni latencia del actuador, y
  peatones con reglas simples. Ver la [ficha](ficha_ambiente.md#simplificaciones).
- **Entrenamiento y reproducibilidad:** el entrenamiento se hace en un computador del equipo (macOS arm64).
  `requirements.lock.txt` fija las versiones; en otra plataforma los resultados pueden variar levemente.

## 8. Aprendizajes

- **Mirar la escena 3D encontró errores que las métricas no mostraban:** el atasco de N3 y el peatón quieto.
- **Cuando ninguna configuración aprende, revisar si el ambiente es justo** antes de subir el presupuesto. Así
  pasó con los conos invisibles y con los pasillos angostos.
- **El currículo y el punto de partida importaron más** que los hiperparámetros.
- **Las semillas importan:** un solo resultado de 97 % escondía una receta que fallaba 2 de 3 veces.
- **La trazabilidad sirve en la práctica:** los modelos de N1 y N2 dejaron de aparecer en la web por un
  cambio de versión. Se encontró y se corrigió porque cada manifiesto guarda con qué ambiente se entrenó.

## 9. Contribuciones

> Completar con el nombre completo de cada integrante y su aporte principal.

| Integrante | Contribución |
|---|---|
| _Nombre completo_ | _…_ |
| _Nombre completo_ | _…_ |

Herramientas: el código y la documentación se desarrollaron con apoyo de un asistente de programación con
IA (Claude Code). Las decisiones, las revisiones y la validación de resultados fueron del equipo.
