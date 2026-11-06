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
| `N1-exp01-dqn` | Una DQN de 2×128 aprende a estacionar en 3000 episodios y supera a la regla (49 %) | baseline aprendido | 3000 episodios (400 803 pasos) · 164 s |
| `N1-exp02-ppo` | PPO (referencia del curso) con 300 000 pasos iguala o supera a la DQN | solo el algoritmo: DQN → PPO | 300 000 pasos (1557 episodios) · 147 s |

Ambos con semilla 42.

### Resultados en `N1-suite-v1`

| Política | Éxito | Choque con obstáculo | Salida del área | Tiempo agotado | Tiempo medio | Error final de posición |
|---|---|---|---|---|---|---|
| Aleatoria | **0 %** | 8 | 14 | 78 | 35.1 s | 9.30 m |
| Regla | **49 %** | 31 | 20 | 0 | 10.0 s | 6.39 m |
| DQN · `N1-exp01-dqn-s42-3ce2f0ec` | **42 %** | 9 | 2 | 47 | 21.8 s | 3.38 m |
| PPO · `N1-exp02-ppo-s42-52969c85` | **0 %** | 0 | 0 | 100 | 40.0 s | 7.04 m |

Gráficas: `results/N1/comparison.png` (resultado por causa) y `results/N1/curves.png` (entrenamiento
frente a validación). Latencia de inferencia media: regla 0.01 ms, DQN 0.012 ms, PPO 0.065 ms por acción.

### Análisis y decisión

- **exp01 · DQN → hipótesis rechazada.** Aprende (de 0 % a 42 %) y choca mucho menos que la regla
  (11 fallos por choque o salida contra 51), pero no la supera en éxito: en 47 episodios se queda sin
  terminar. Es más prudente pero menos decidida que la regla.
- **exp02 · PPO → hipótesis rechazada.** No consiguió ni un éxito durante el entrenamiento (curva de
  entrenamiento y de validación en 0 %). Sin éxitos no hay señal que aprender: convergió a moverse sin
  chocar hasta agotar el tiempo. Es un problema de exploración con este presupuesto.
- **Modelo recomendado del N1: DQN `N1-exp01-dqn-s42-3ce2f0ec`**, el mejor modelo aprendido. La regla
  sigue siendo la mejor política del nivel y queda como referencia a superar.
- Nota de trazabilidad: el run de PPO registra `git_dirty: true` porque mientras entrenaba se agregó la
  función `plot_curves` a `pipelines/evaluate.py`. El cambio solo añade una gráfica; no modifica el
  entrenamiento ni la evaluación (`run_case`, `evaluate`).

### Siguientes hipótesis razonables

1. PPO con más presupuesto (1–2 millones de pasos) y mayor entropía (`ent_coef` 0.05) para que explore
   lo suficiente como para encontrar éxitos.
2. DQN con más episodios o una penalización mayor por agotar el tiempo, para convertir esos 47 episodios
   sin terminar en intentos de estacionar.
3. Tres semillas para los dos finalistas (DQN y el mejor PPO), como pide la guía.
