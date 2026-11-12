# Ficha del ambiente · Autonomous Parking

| Campo | Valor |
|---|---|
| Versión del contrato | `parking-v8` (`envs/parking_env.py: ENV_VERSION`); cada nivel declara la versión que usa (N1: `parking-v2`, N2: `parking-v3`, N3 y N4: `parking-v6`, N5: `parking-v8`) |
| Implementación | `envs.parking_env.ParkingEnv` (Gymnasium 1.x) |
| Misión | Llevar el vehículo al cupo objetivo y detenerlo alineado, sin chocar |
| Unidades | metros, segundos, radianes, m/s |

## Mundo

| Elemento | Valor |
|---|---|
| Parqueadero | 30 × 20 m, bordes en x ∈ [−15, 15], y ∈ [−10, 10] |
| Cupos | 2 filas × 8 cupos de 3.0 × 5.5 m; pasillo central de 9 m |
| Vehículo | 4.2 × 1.8 m, distancia entre ejes 2.6 m |
| Obstáculos | autos estacionados (cada cupo ocupado con p = 0.6; p = 0.85 los vecinos del objetivo), 4 columnas de 0.8 × 0.8 m, conos de 0.5 × 0.5 m. Con `blocking_cones = 1` (N2) se agrega un cono extra en la línea recta entre el arranque y el cupo, a ≥ 3.2 m del auto y ≥ 1.7 m de la entrada (se omite si no cabe: 67 de 100 episodios de la suite lo tienen) |
| Peatones (opcional, N3) | cajas de 0.5 × 0.5 m que cruzan el pasillo en una x fija, ida y vuelta entre y = ±3.9 m, a 0.8–1.3 m/s; se detienen si el auto está a menos de 0.6 m de su camino; los sensores los detectan. Con `crossing_pedestrians = 1` el primero cruza entre el arranque y el cupo (a ≥ 3.8 m del auto), empezando a 1.5–3.5 m del carril y caminando hacia él |
| Mapa garage (N5, `layout: garage`) | azotea de 40 × 50 m (x ∈ [−20, 20], y ∈ [−25, 25]) con 3 filas (A, B, C) de 8 cupos de 3.0 × 5.5 m en \|x\| ≤ 12, todas abiertas hacia abajo, cada una frente a un pasillo de 8.5 m; bordillos de 0.3 m detrás de las filas B y C (no se puede cruzar por un cupo vacío); pasillos laterales de 7.7 m (\|x\| > 12.3) que conectan todo; entrada abajo en x = 0. Cada cupo no objetivo ocupado con p = 0.85. Sin columnas, conos ni peatones. El auto arranca en la entrada (0, −22) mirando hacia adentro |
| Aleatoriedad por episodio | cupo objetivo (16 posibles), ocupación de cupos, desplazamiento ±0.15 m de los autos, pose inicial y conos; reproducible con `reset(seed=...)` |

## Dinámica

- Modelo cinemático de bicicleta:
  `x += v·cos θ·h`, `y += v·sin θ·h`, `θ += v/L·tan δ·h`.
- **Paso físico** `h = 0.1 s` (2 subpasos por decisión). **Frecuencia de decisión** 5 Hz (`dt = 0.2 s`).
- Velocidad en [−2, 3] m/s, con rodadura que la multiplica por 0.97 en cada paso. Dirección δ en [−0.6, 0.6] rad; se mantiene entre pasos, como un volante.
- La colisión se revisa en cada subpaso con el teorema del eje separador (cajas orientadas).
- La velocidad de animación de la web **no** modifica `dt` ni las reglas.

## Acciones · `Discrete(5)`

| # | Acción | Efecto por decisión |
|---|---|---|
| 0 | Avanzar | v += 0.4 m/s |
| 1 | Retroceder | v −= 0.4 m/s |
| 2 | Izquierda | δ += 0.15 rad |
| 3 | Derecha | δ −= 0.15 rad |
| 4 | Frenar | \|v\| −= 1.0 m/s, hasta 0 |

## Observación · `Box(14,)` float32 (N1) · `Box(19,)` (N2–N4) · `Box(25,)` (N5)

| Índice | Variable | Unidad / escala | Rango (con clipping) |
|---|---|---|---|
| 0 | dx del cupo en el marco del vehículo (adelante) | m / 10 | [−5, 5] |
| 1 | dy del cupo en el marco del vehículo (izquierda) | m / 10 | [−5, 5] |
| 2 | error de orientación, plegado a ±90° | rad / (π/2) | [−1, 1] |
| 3 | velocidad | (m/s) / 3 | [−1, 1] |
| 4 | ángulo de dirección | rad / 0.6 | [−1, 1] |
| 5 | las 4 esquinas están dentro del cupo | 0 o 1 | {0, 1} |
| 6–13 | 8 sensores de distancia cada 45° (0 = frente, sentido antihorario), medidos desde la carrocería | m / 8 | [0, 1] |
| 14–18 | **solo con `bumper_sensors = 5` (N2):** rayos paralelos hacia adelante desde 5 puntos repartidos a lo ancho del parachoques delantero (y = −0.9 … 0.9 m) | m / 4 | [0, 1] |
| 19–20 | **solo en el mapa garage (N5):** posición del auto en el mapa (x / 20, y / 25) | — | [−1, 1] |
| 21–22 | **solo N5:** rumbo del auto (cos θ, sin θ) | — | [−1, 1] |
| 23–24 | **solo N5:** posición del cupo libre en el mapa (x / 20, y / 25) | — | [−1, 1] |

**Qué sabe el auto en N5 y qué no.** Ve dónde está él y dónde está el cupo libre (como un tablero que muestra el
espacio disponible), pero **no recibe la ruta**: tiene que aprender a rodear las filas. La ruta (Dijkstra sobre
una cuadrícula de 0.5 m, `envs/garage.py: RouteField`) solo se usa en la recompensa.

**Por qué N2 agrega sensores de parachoques.** Con los 8 rayos radiales, un cono de 0.5 m puede quedar
entre dos rayos: en 36 de los 67 episodios con cono en el camino, el auto chocaba sin que ningún sensor lo
detectara. Con los 5 rayos del parachoques lo detecta en los 67. Como cambia la observación (14 → 19), los
modelos de N2 son una **familia nueva**: no pueden partir de un checkpoint de N1, y un modelo solo se sirve
en la definición de nivel con la que se entrenó (`envs.levels.model_matches_level`).

El error de orientación se pliega para que entrar de frente o en reversa sean igual de válidos.
Con `sensor_noise = σ > 0` se suma ruido N(0, σ) a los sensores y N(0, σ/2) a dx y dy.

## Recompensa

```
r = −0.05                                    tiempo
  + Φ(s') − Φ(s)                             acercamiento y alineación (shaping por potencial)
  − 0.3·|v|          si está dentro del cupo
  − 0.15             si |v| < 0.1 fuera del cupo (evita quedarse quieto)
  − 50               colisión
  + 100              estacionado
Φ(s) = −d − 1.5 · max(0, 1 − d/6) · |error de orientación|        d = distancia al centro del cupo (m)
```

**N5 (garage).** `d` es la distancia **por la ruta** (no en línea recta): rodear las filas también acerca. El costo
de cada tramo es mayor junto a las filas y los bordes (×1 a ×4), así la ruta va por el centro de los pasillos; a
menos de 4 m del cupo pasa suavemente a la distancia en línea recta. Además, detenerse dentro del cupo sin quedar
estacionado (torcido o a medias) también cuesta −0.15 por paso.

## Terminación

| Señal | Condición |
|---|---|
| `terminated` · éxito | las 4 esquinas dentro del cupo (holgura de 0.15 m), \|error de orientación\| ≤ 0.25 rad (14°) y \|v\| ≤ 0.4 m/s |
| `terminated` · fallo | choque con un obstáculo (`collision_kind = "obstacle"`), atropello de un peatón (`"pedestrian"`) o salida del parqueadero (`"boundary"`) |
| `truncated` | 200 decisiones (40 s simulados) sin terminar; en N5, 400 (80 s) |

`info` contiene: `collision_kind`, `distance`, `heading_error`, `speed`, `steer`, `collision`, `success`, `in_slot`, `steps`, `target_id`.

## Niveles (`configs/levels/*.json`)

Cada nivel se desarrolla en su propia rama (`feature/nivel-N`), se integra a `main` y se marca con un
tag. Los niveles anteriores siguen disponibles.

| Nivel | Estado | Configuración | Qué decisión nueva debe tomar el agente |
|---|---|---|---|
| N1 · Misión base | ✅ `nivel-1` | arranca en el pasillo a 4–8 m del cupo, rodeado de autos estacionados; 0 conos, 0 peatones, sin ruido | girar 90° hacia el cupo sin rozar a los vecinos y detenerse alineado |
| N2 · Conos | ✅ `nivel-2` | N1 + 3 conos sueltos + 1 cono en la ruta (`difficulty: hard`, `blocking_cones: 1`) + 5 sensores de parachoques | rodear un obstáculo fijo que tapa el camino directo |
| N3 · Peatones | ✅ `nivel-3` | N1 + 2 peatones, uno cruza la ruta (`crossing_pedestrians: 1`, `pedestrians_yield`) + sensores de parachoques | frenar y ceder el paso a un obstáculo en movimiento, y retomar (perturbación con recuperación) |
| N4 · Todo junto | ✅ `nivel-4` | N2 + N3 (3 conos + cono en la ruta, 2 peatones, uno cruza) + ruido gaussiano σ = 0.1 m en los sensores y 0.05 m en la posición del cupo | combinar todo con sensores imprecisos (perturbación) |
| N5 · Garaje grande | ✅ `nivel-5` | mapa garage (3 filas, entrada abajo) + sensores de parachoques + ruido σ = 0.1 m; 400 pasos | planear el recorrido: rodear las filas por el pasillo lateral correcto y estacionar al final |

Todos los niveles comparten las acciones. N1 observa 14 valores, N2–N4 19 y N5 25: son tres familias de
modelos. Dentro de una familia, un modelo antiguo puede evaluarse en un nivel nuevo, pero no se espera que
lo resuelva.

**Dificultad `tutorial`.** Solo para etapas de currículo de entrenamiento (nunca para evaluar): el auto
arranca con la nariz ya dentro del cupo, de modo que la exploración al azar estaciona en ~8 % de los
episodios (frente a ~0.2 % desde el pasillo).

## Historial de versiones

| Versión | Cambio |
|---|---|
| `parking-v1` | Contrato inicial: cupos, autos estacionados, columnas, conos y ruido opcional |
| `parking-v2` | Agrega peatones en movimiento (solo cuando `pedestrians > 0`, es decir en N3) y `info.collision_kind`. N1 y N2 producen exactamente las mismas transiciones que en v1 (lo verifica `test_levels_without_pedestrians_are_unchanged`) |

| `parking-v3` | Agrega opciones de nivel: `blocking_cones`, `crossing_pedestrians`, `bumper_sensors` y la dificultad `tutorial`. Con sus valores por defecto reproduce v2 exactamente, por eso N1 sigue declarando `parking-v2` (`SUPPORTED_VERSIONS`). |
| `parking-v4` | Agrega `pedestrians_yield`: si el auto está quieto (|v| ≤ 0.1 m/s) el peatón sigue caminando, o se devuelve si no cabe. En v3 el auto y el peatón podían quedarse esperándose hasta el timeout (4 de 100 episodios en N3). Sin la opción reproduce v3 exactamente; solo N3 la usa. |
| `parking-v5` | Si hay a la vez conos en el camino (`blocking_cones`) y peatón que cruza, el peatón cruza 1.5 m más allá del cono que tenga en su línea (camina todo el pasillo y no debe atravesar conos); si no cabe antes del cupo se omite. Sin esa combinación reproduce v4 exactamente; solo N4 la usa. |
| `parking-v6` | Agrega `pedestrians_evade`: los peatones **nunca se quedan quietos**. Si el auto en movimiento está a menos de 0.6 m de su camino y avanzar lo acerca, el peatón da la vuelta y se aleja; si el auto le queda encima, se aparta en la dirección opuesta; si queda arrinconado contra el borde del pasillo, se hace a un lado a lo largo de la fila. Con el auto quieto sigue cruzando. El estado incluye el rumbo del peatón (`heading`) para dibujarlo mirando hacia donde camina. N3 y N4 la usan. |
| `parking-v7` | Mapa `garage` con pasillos de 6.7 m. Solo lo usaron 4 pilotos de N5; **ya no se reproduce** (v8 cambió la geometría), por eso esos modelos aparecen como «otra versión del nivel». |
| `parking-v8` | Mapa `garage` (`layout`) con pasillos de 8.5 m, dificultad `roam` y `route_start` (arranques sobre la ruta desde la entrada, a [desde, hasta] m del cupo) para el currículo inverso, potencial por la ruta y castigo por detenerse a medias dentro del cupo. Sin `layout: garage` reproduce v6 exactamente. Solo N5 lo usa. |

## Simplificaciones

- Movimiento en el plano; sin derrape, inercia lateral ni retardo de actuadores.
- Acciones discretas: el agente no elige aceleraciones continuas.
- Los peatones cruzan el pasillo en línea recta y nunca se detienen: si el auto se acerca se alejan o se apartan (v6); los autos estacionados no se mueven.
- Sensores ideales salvo el ruido gaussiano opcional; sin oclusiones parciales ni fallas.
- La visualización 3D (Three.js) solo representa el estado que entrega Python; no tiene física propia.
