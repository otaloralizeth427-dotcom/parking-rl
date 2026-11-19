# Modelo servido, rollback y monitoreo

## Modelo servido por nivel

Cada nivel tiene un archivo `serving/<NIVEL>.json`, versionado en git, que indica qué run se sirve (`current`).
Su historial solo crece: cada promoción o rollback agrega una entrada con el run, la acción, el motivo, la
versión del ambiente, el hash del modelo, el éxito en la suite y la fecha.

```bash
python -m agents.serving status   N5
python -m agents.serving promote  N5 <run_id> --reason "…"
python -m agents.serving rollback N5 --reason "…"
```

Antes de mover el puntero se comprueban cuatro cosas:
- el run es del nivel;
- está completo;
- es compatible con la versión del ambiente (`model_matches_level`);
- su modelo carga con el hash del manifiesto.

**Cómo llega a producción.** El servidor toma el agente por defecto de cada partida nueva desde ese puntero.
Si un nivel no tiene puntero, usa el de mayor éxito en la suite. Como el puntero está en git, el cambio llega a
producción así:
1. commit;
2. push a `main`;
3. Render vuelve a publicar;
4. `GET /api/health` muestra el run servido de cada nivel y el commit, para comprobarlo.

**Desde la web (panel → «Modelo servido»):**
- muestra el modelo servido con su hash;
- muestra el historial de versiones;
- tiene dos botones: «Usar el servido» y «Volver a la versión anterior». Este último hace el rollback en la
  propia partida, para revisarlo sin tocar el puntero oficial.

## Prueba de rollback (simulacro del 19-11-2026, nivel N5)

| Paso | Acción | Modelo servido | Comprobación |
|---|---|---|---|
| 0 | Estado inicial | `N5-exp05-ppo-inverso-3m-s44-78c82696` (sha256 `68902bb9…`) | — |
| 1 | `promote` del candidato `N5-exp06-ppo-inverso-3m-g995-s43-bb22f4d9` | candidato (sha256 `03583f45…`) | `/api/health` y una partida nueva por WebSocket lo usan |
| 2 | Comparación en el servidor: mismos 30 casos de `N5-suite-v1` | — | exp05: **100 %, 14.7 s** · candidato: **100 %, 15.3 s** |
| 3 | `rollback` porque empata en éxito pero es más lento | vuelve `N5-exp05-…-78c82696` (mismo sha256 `68902bb9…`) | `/api/health` y una partida nueva vuelven al anterior |

El historial completo, con los motivos y la hora de cada paso, está en [`serving/N5.json`](../serving/N5.json).
El comportamiento también queda cubierto por pruebas automáticas:
- `test_promote_and_rollback_keep_an_append_only_history`
- `test_server_serves_the_pointer_and_rollback_changes_it`
- `test_promote_rejects_incompatible_or_unknown_runs`

## Evaluar y comparar desde la web

En el panel, «Evaluar y comparar» permite elegir de 1 a 4 agentes (modelos históricos, el servido o las
referencias) y de 1 a 30 casos.
- **Mismos casos para todos.** El servidor corre a cada agente sobre los primeros casos de la suite fija
  del nivel, con las mismas semillas.
- **Mismo código que el pipeline** (`pipelines/scoring.py`). La prueba `test_evaluate_from_the_web_matches_the_pipeline`
  comprueba que dan los mismos resultados caso por caso.
- **Resultado:** una tabla con % de éxito, tiempo medio y causas de fallo. Al pasar el mouse sobre cada fila
  se ven el hash y el commit del modelo.

## Monitoreo

`GET /api/metrics` (y el panel → «Monitoreo del servidor») reporta, desde que arrancó el servidor:

- **Por nivel y por agente:** episodios terminados, % de éxito y causas de fallo (obstáculo, peatón, salida
  del área, tiempo agotado). Los episodios manejados a mano no se cuentan.
- **Latencia por paso del ambiente:** media y p95 de los últimos 5 000 pasos.
- **Operación:** partidas activas, commit servido y modelo servido de cada nivel.

Las cifras viven en memoria. Si el servidor gratuito se reinicia o se duerme, vuelven a cero, y eso está
documentado como limitación. El monitoreo de calidad con persistencia es el del pipeline: la suite fija y el
test final de cada nivel, guardados en `results/`.
