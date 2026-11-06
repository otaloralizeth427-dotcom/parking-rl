# Ficha del ambiente · Autonomous Parking

| Campo | Valor |
|---|---|
| Versión del contrato | `parking-v1` (`envs/parking_env.py: ENV_VERSION`) |
| Implementación | `envs.parking_env.ParkingEnv` (Gymnasium 1.x) |
| Misión | Llevar el vehículo al cupo objetivo y detenerlo alineado, sin chocar |
| Unidades | metros, segundos, radianes, m/s |

## Mundo

| Elemento | Valor |
|---|---|
| Parqueadero | 30 × 20 m, bordes en x ∈ [−15, 15], y ∈ [−10, 10] |
| Cupos | 2 filas × 8 cupos de 3.0 × 5.5 m; pasillo central de 9 m |
| Vehículo | 4.2 × 1.8 m, distancia entre ejes 2.6 m |
| Obstáculos | autos estacionados (cada cupo ocupado con p = 0.6; p = 0.85 los vecinos del objetivo), 4 columnas de 0.8 × 0.8 m, conos de 0.5 × 0.5 m |
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

## Observación · `Box(14,)` float32

| Índice | Variable | Unidad / escala | Rango (con clipping) |
|---|---|---|---|
| 0 | dx del cupo en el marco del vehículo (adelante) | m / 10 | [−5, 5] |
| 1 | dy del cupo en el marco del vehículo (izquierda) | m / 10 | [−5, 5] |
| 2 | error de orientación, plegado a ±90° | rad / (π/2) | [−1, 1] |
| 3 | velocidad | (m/s) / 3 | [−1, 1] |
| 4 | ángulo de dirección | rad / 0.6 | [−1, 1] |
| 5 | las 4 esquinas están dentro del cupo | 0 o 1 | {0, 1} |
| 6–13 | 8 sensores de distancia cada 45° (0 = frente, sentido antihorario), medidos desde la carrocería | m / 8 | [0, 1] |

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

## Terminación

| Señal | Condición |
|---|---|
| `terminated` · éxito | las 4 esquinas dentro del cupo (holgura de 0.15 m), \|error de orientación\| ≤ 0.25 rad (14°) y \|v\| ≤ 0.4 m/s |
| `terminated` · fallo | choque con un auto, cono, columna o bordillo, o salida del parqueadero |
| `truncated` | 200 decisiones (40 s simulados) sin terminar |

`info` contiene: `distance`, `heading_error`, `speed`, `steer`, `collision`, `success`, `in_slot`, `steps`, `target_id`.

## Niveles (`configs/levels/*.json`)

| Nivel | Dificultad | Ruido | Qué decisión nueva debe tomar el agente |
|---|---|---|---|
| N1 · Misión base | `easy`: arranca frente al cupo (±1 m, ±17°), sin conos | 0 | entrar recto y frenar dentro del cupo |
| N2 · Más decisiones | `normal`: arranca en el pasillo a 4–8 m del cupo, 2 conos fuera de la línea directa | 0 | planear el giro de 90° sin rozar a los vecinos |
| N3 · Operación exigente | `hard`: como N2 con 5 conos que pueden bloquear la ruta | σ = 0.1 m | esquivar obstáculos con sensores imprecisos |

Los tres niveles comparten observaciones y acciones, así que sus modelos forman una sola familia compatible (`parking-v1`).

## Simplificaciones

- Movimiento en el plano; sin derrape, inercia lateral ni retardo de actuadores.
- Acciones discretas: el agente no elige aceleraciones continuas.
- Obstáculos estáticos durante el episodio; no hay peatones ni autos en movimiento.
- Sensores ideales salvo el ruido gaussiano opcional; sin oclusiones parciales ni fallas.
- La visualización 3D (Three.js) solo representa el estado que entrega Python; no tiene física propia.
