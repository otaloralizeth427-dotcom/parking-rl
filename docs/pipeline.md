# Pipeline (Prefect)

`pipelines/flow.py` une en un solo flujo de Prefect los pasos que antes se corrían a mano.

```bash
# Un checkpoint ya registrado compite contra el modelo servido
python -m pipelines.flow --level N5 --run-id N5-exp06-ppo-inverso-3m-g995-s42-c8fe1d42
# Entrenar un run nuevo y llevarlo hasta producción si gana
python -m pipelines.flow --level N5 --experiment configs/experiments/N5-exp05-ppo-inverso-3m.json --seed 45
# Decidir sin mover el puntero
python -m pipelines.flow ... --dry-run
```

| # | Paso | Qué comprueba o produce |
|---|---|---|
| 1 | **Validar** | El nivel existe y su versión del ambiente está soportada. El experimento es del nivel y pasa `resolve`: presupuesto, currículo y suite. El checkpoint es compatible con la versión del nivel. |
| 2 | **Pruebas** | `pytest` (las rápidas por defecto; todas con `--full-tests`). Si alguna falla, el flujo se detiene. |
| 3 | **Entrenar o cargar** | Con `--experiment` llama a `run_experiment`: run en MLflow, manifiesto y paquete en `artifacts/`. Con `--run-id` carga el checkpoint registrado. |
| 4 | **Evaluar** | Candidato y modelo servido se evalúan sobre los **mismos 100 casos** de la suite fija. El resultado por caso queda en `results/<nivel>/pipeline/<fecha>/eval_*.json`. |
| 5 | **Registrar y verificar hashes** | El sha256 del modelo coincide con el manifiesto. La configuración no cambió después del run (`config_sha256`). El run de MLflow existe y corresponde al manifiesto. |
| 6 | **Criterio de selección** | Se promueve si el candidato tiene **más éxito** en la suite, o **el mismo éxito y es al menos 2 % más rápido**. En cualquier otro caso se rechaza, y el motivo queda escrito. |
| 7 | **Publicar** | Mueve el puntero `serving/<nivel>.json` con `promote` y deja una entrada en el historial. Llega a producción con commit y push a `main`, que Render vuelve a publicar. |
| 8 | **Verificar versión servida** | Una partida nueva del servidor debe usar el puntero esperado y terminar episodios. Si no, hace **rollback automático** al anterior. |

Cada ejecución deja:
- `results/<nivel>/pipeline/<fecha>/report.json`: el reporte completo, con las evaluaciones, los hashes verificados, la decisión y la verificación;
- una línea en `results/<nivel>/pipeline/decisiones.jsonl`: promovido, rechazado, simulado o revertido.

## Ejecuciones registradas

| Fecha (UTC) | Nivel | Candidato | Resultado | Motivo |
|---|---|---|---|---|
| 2026-11-19 00:20 | N5 | `N5-exp06-ppo-inverso-3m-g995-s42` | **rechazado** | mismo éxito (100 %), pero más lento: 14.91 s contra 14.45 s |
| 2026-11-19 00:21 | N5 | `N5-exp05-ppo-inverso-3m-s43` | **rechazado** | mismo éxito (100 %), pero no 2 % más rápido: 14.58 s contra 14.45 s |

El rollback automático del paso 8 está cubierto por `test_pipeline_verification_rolls_back_a_broken_promotion`.
El rollback manual, con su simulacro real, está en [rollback_y_monitoreo.md](rollback_y_monitoreo.md).

## Integración continua

`.github/workflows/ci.yml` corre todas las pruebas y construye la web en cada push.
Al llegar a `main`:
- **Render** vuelve a publicar el servidor (`render.yaml`);
- **Vercel** vuelve a publicar la web.
