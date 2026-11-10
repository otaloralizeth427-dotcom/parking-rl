# Registro de experimentos

Cada run tiene su manifiesto en `manifests/<run_id>.json`, su paquete en `artifacts/<nivel>/<run_id>/`
(pesos, configuración resuelta, curvas y evaluación por episodio) y su ejecución en MLflow
(`mlflow ui --backend-store-uri sqlite:///mlflow.db`). Se conservan también los intentos que no mejoraron.

Métrica principal: **% de éxito en la suite fija del nivel** (modelo congelado, sin exploración).
El retorno solo se compara dentro de un mismo nivel y la misma función de recompensa.

---

## Nivel 1 · Misión base (`configs/levels/N1.json`)

El vehículo arranca en el pasillo, a 4–8 m del cupo, rodeado de autos estacionados. Sin conos, sin
peatones y sin ruido. Suite: `N1-suite-v1` (100 episodios, semillas 100000–100099).
Código: commit `4d94880` (rama `feature/nivel-1`), ambiente `parking-v2`.

### Referencias

| Política | Qué es |
|---|---|
| Aleatoria | acción uniforme en cada paso |
| Regla | seguimiento geométrico del eje del cupo (`agents/rule_based.py`), sin aprendizaje |

### Experimentos

| Experimento | Hipótesis | Cambio frente a la referencia | Presupuesto |
|---|---|---|---|
| `N1-exp01-dqn` | Una DQN de 2×128 aprende a estacionar en 3000 episodios y supera a la regla (49 %) | baseline aprendido | 3000 episodios · ~3 min |
| `N1-exp02-ppo` | PPO (referencia del curso) con 300 000 pasos iguala o supera a la DQN | solo el algoritmo: DQN → PPO | 300 000 pasos · ~2.5 min |
| `N1-exp03-ppo` | PPO falló por falta de exploración: con 1 M de pasos y `ent_coef` 0.05 encuentra éxitos | exp02 con más presupuesto y más entropía | 1 000 000 pasos · ~8 min |
| `N1-exp04-dqn` | La DQN seguía mejorando: con el doble de episodios supera a la regla | exp01 con 6000 episodios y `epsilon_decay` 0.999 | 6000 episodios · ~6 min |

Primera ronda con semilla 42. Los dos finalistas (exp03 y exp01) se repitieron con las semillas 43 y 44
(`python -m pipelines.train <config> --seed 43`). Código de todos los runs de esta ronda: commit `7f7a574`
en `feature/nivel-1`, sin cambios locales (`git_dirty: false`).

### Resultados en `N1-suite-v1` (100 episodios, modelo congelado)

| Política | Éxito | Choque | Salida | Tiempo agotado | Tiempo medio | Error final |
|---|---|---|---|---|---|---|
| Aleatoria | **0 %** | 8 | 14 | 78 | 35.1 s | 9.30 m |
| Regla | **49 %** | 31 | 20 | 0 | 10.0 s | 6.39 m |
| exp01 DQN · s42 `3ce2f0ec` | **42 %** | 9 | 2 | 47 | 21.8 s | 3.38 m |
| exp01 DQN · s43 `c37b531a` | 0 % | 31 | 8 | 61 | 27.0 s | 13.63 m |
| exp01 DQN · s44 `efdbe060` | 8 % | 12 | 4 | 76 | 33.1 s | 7.67 m |
| exp02 PPO · s42 `52969c85` | 0 % | 0 | 0 | 100 | 40.0 s | 7.04 m |
| **exp03 PPO · s42 `64bf7433`** | **97 %** | 0 | 0 | 3 | 6.6 s | 0.72 m |
| exp03 PPO · s43 `422ab16e` | 0 % | 0 | 0 | 100 | 40.0 s | 6.41 m |
| exp03 PPO · s44 `aade6f7d` | 0 % | 4 | 0 | 96 | 39.3 s | 5.43 m |
| exp04 DQN · s42 `2216c023` | 0 % | 0 | 10 | 90 | 36.5 s | 11.47 m |

**Variación por semilla (finalistas):**

| Receta | s42 | s43 | s44 | Media ± desv. |
|---|---|---|---|---|
| exp03 PPO | 97 % | 0 % | 0 % | 32 % ± 46 |
| exp01 DQN | 42 % | 0 % | 8 % | 17 % ± 18 |

Gráficas: `results/N1/comparison.png` y `results/N1/curves.png` (entrenamiento frente a validación por run).

### Test final reservado (`N1-final-v1`)

100 episodios cuyas semillas **no están en el repo** (viven en `~/.parking-rl/`); el repo solo guarda su
huella SHA-256 (`configs/suites/N1-final-v1.json`), así se puede verificar que el test no cambió. Se usó
una sola vez, con los modelos elegidos y la regla (`results/N1/final_test.json`):

| Política | Éxito | Choque | Salida | Tiempo agotado |
|---|---|---|---|---|
| Regla | 53 % | 31 | 16 | 0 |
| **exp03 PPO · s42** | **97 %** | 0 | 0 | 3 |
| exp01 DQN · s42 | 48 % | 13 | 0 | 39 |

El 97 % se mantiene en casos nunca vistos: el modelo no memorizó la suite de selección.

### Análisis y decisión

- **exp01 · DQN → rechazada.** 42 % con s42, pero 0 % y 8 % con las otras semillas: no supera a la regla.
- **exp02 · PPO → rechazada.** Sin éxitos durante el entrenamiento, no hay señal que aprender.
- **exp03 · PPO → aceptada para el modelo, rechazada para la receta.** Con s42 llega a 97 % (suite y test
  final), choca 0 veces y estaciona en 6.6 s. Pero con s43 y s44 se queda en 0 %: el éxito depende de que
  la exploración encuentre el cupo a tiempo. El modelo s42 es bueno; la receta no es confiable todavía.
- **exp04 · DQN → rechazada.** Con exploración más lenta nunca encontró éxitos (0 % en todo el entrenamiento).
- **Modelo servido en N1: PPO `N1-exp03-ppo-s42-64bf7433`.** La web lo carga por defecto (mayor éxito en la
  suite) y verifica su hash antes de usarlo. Referencia a superar en los siguientes niveles: la regla.
- Nota de trazabilidad: el run `N1-exp02-ppo` registra `git_dirty: true` porque mientras entrenaba se agregó
  `plot_curves` a `pipelines/evaluate.py` (solo una gráfica; no cambia entrenamiento ni evaluación).

### Siguientes hipótesis razonables

1. Hacer confiable la receta: currículo (empezar más cerca del cupo) o más entropía al inicio, y repetir
   con 3 semillas. Éxito = las 3 semillas por encima de la regla.
2. Usar el PPO s42 como punto de partida (`parent_run_id`) para los niveles 2 a 4 en lugar de entrenar desde cero.
