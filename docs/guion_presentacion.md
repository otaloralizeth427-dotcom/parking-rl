# Guion de la presentación (10 minutos)

**Página:** `presentacion.html` en el sitio del proyecto (https://parking-rl.vercel.app/presentacion.html).
Se avanza con las flechas del teclado.

**Reparto sugerido.** Lizeth abre y explica el problema, el ambiente y los niveles. Esteban explica el modelo,
los resultados y lo que no funcionó. Lizeth cuenta el versionamiento y la operación. Los dos hacen la demo y el
cierre. Se puede cambiar, siempre que cada uno pueda explicar todo.

| # | Diapositiva | Quién | Tiempo |
|---|---|---|---|
| 1 | Portada | Lizeth | 0:20 |
| 2 | El problema | Lizeth | 0:45 |
| 3 | El ambiente | Lizeth | 1:00 |
| 4 | Seis niveles | Lizeth | 0:45 |
| 5 | El modelo | Esteban | 1:00 |
| 6 | Resultados | Esteban | 0:45 |
| 7 | Lo que no funcionó | Esteban | 1:00 |
| 8 | Versionamiento | Lizeth | 1:00 |
| 9 | Pipeline y despliegue | Lizeth | 0:45 |
| 10 | Demo | Los dos | 2:00 |
| 11 | Limitaciones y aprendizajes | Esteban | 0:30 |
| 12 | Cierre | Los dos | 0:10 |

---

## 1 · Portada (Lizeth, 20 s)

> Buenas. Somos Lizeth Otálora y Esteban Molina, del grupo A. Nuestro escenario es *Autonomous Parking*: un
> auto que aprende solo a estacionarse, con todo el proceso de MLOps alrededor, desde la configuración hasta el
> modelo que ven en la web.

## 2 · El problema (Lizeth, 45 s)

> La misión es simple de decir y difícil de lograr: llevar el auto hasta un cupo libre y dejarlo adentro,
> alineado y detenido, sin tocar otros autos, conos ni personas, y sin salirse.
> ¿Por qué importa? Estacionar causa muchos roces y pérdida de tiempo. Un auto que lo hace solo aprovecha
> mejor el espacio y protege a quien camina por el parqueadero.
> Lo resolvimos con aprendizaje por refuerzo: el auto prueba, recibe premios y castigos, y mejora.

## 3 · El ambiente (Lizeth, 1 min)

> El ambiente lo construimos en Gymnasium.
> - **Lo que el auto observa:** dónde está el cupo respecto a él, qué tan torcido va, su velocidad y dirección,
>   y lo que miden sus sensores. Son 8 sensores tipo LiDAR más 5 en el parachoques para ver conos pequeños.
> - **Las acciones:** son 5, avanzar, retroceder, girar a la izquierda, girar a la derecha y frenar, con una
>   física de bicicleta que avanza un paso cada 0,2 segundos.
> - **La recompensa:** +100 si estaciona y −50 si choca. En el camino gana un poco por acercarse y alinearse, y
>   pierde un poco por el tiempo y por quedarse quieto.
> - **El final del episodio:** se *termina* si estaciona o choca, y se *trunca* si se acaba el tiempo.

## 4 · Seis niveles (Lizeth, 45 s)

> Diseñamos seis niveles, y cada uno agrega una decisión nueva:
> - el pasillo básico;
> - conos que hay que rodear;
> - peatones que cruzan;
> - todo junto con sensores imprecisos;
> - un garaje grande donde hay que buscar la ruta rodeando las filas;
> - y estacionar en reversa.
>
> Cada nivel declara su versión del ambiente. Fíjense que van de la v2 a la v14: cada vez que cambiamos cómo
> funciona el mundo, subimos la versión.

## 5 · El modelo (Esteban, 1 min)

> Usamos PPO de Stable-Baselines3, el algoritmo de referencia del curso. También probamos DQN y Q-learning,
> pero no superaron a una regla simple.
> Lo que más ayudó fue el **currículo**: primero el auto arranca cerca del cupo y luego cada vez más lejos.
> Sin eso, dependía de encontrar un éxito por suerte.
> En los niveles 3 y 4 continuamos desde el mejor modelo anterior y obtuvimos lo mismo con un tercio del
> entrenamiento. En el garaje usamos un currículo al revés, desde el cupo hacia la entrada.
> Cada receta la entrenamos con 3 semillas, porque un resultado de 97 % con una sola semilla escondía una
> receta que fallaba dos de cada tres veces.

## 6 · Resultados (Esteban, 45 s)

> Evaluamos siempre con el modelo congelado, en 100 episodios fijos. El auto al azar no estaciona nunca, y una
> regla escrita a mano llega entre 16 y 49 %. La IA llega a 91–100 % según el nivel.
> Elegimos el modelo con la prueba fija y lo confirmamos **una sola vez** con un examen final cuyas semillas no
> están en el repositorio. Por eso esos números son honestos: el modelo nunca vio esos casos.

## 7 · Lo que no funcionó (Esteban, 1 min)

> Esta diapositiva es la que más nos enseñó. Varias veces nada aprendía, y el problema no era el modelo sino el
> ambiente:
> - **Conos invisibles:** el cono cabía entre los rayos de los sensores y en la mitad de los casos el auto no lo
>   veía. Agregamos sensores en el parachoques.
> - **Conos que tapaban el giro:** con un cono cerca del cupo solo estacionaba 58 %. Hicimos los conos
>   rodeables y el pasillo más ancho, y pasamos de 80 a 96 %.
> - **Atropellos:** el modelo veía al peatón y seguía acelerando. Agregamos un freno de emergencia, como los de
>   los autos reales, y los atropellos bajaron a cero.
> - **Garaje con pasillos angostos:** los primeros cuatro intentos se quedaron entre 0 y 12 %, hasta que
>   ensanchamos los pasillos.

## 8 · Versionamiento (Lizeth, 1 min)

> Cada modelo sabe de dónde viene. El camino es este:
> 1. Una configuración en JSON con su hipótesis.
> 2. El entrenamiento, que registra el commit y la semilla.
> 3. MLflow guarda los parámetros, las métricas y el paquete.
> 4. Un manifiesto guarda el SHA-256 del modelo y de la configuración.
> 5. Al cargar un modelo verificamos ese hash; si alguien cambiara el archivo, no carga.
> 6. Luego viene la evaluación y, al final, un puntero en git dice qué modelo se sirve, con historial.
>
> Tenemos 35 configuraciones con hipótesis, más de 80 runs en MLflow (incluidos los que fallaron), ramas por
> nivel y tags.

## 9 · Pipeline y despliegue (Lizeth, 45 s)

> Todo eso lo une un pipeline de 8 pasos en Prefect. Solo promueve un modelo si gana en la prueba fija; si al
> publicarlo algo falla, vuelve solo a la versión anterior.
> La web está en Vercel, hecha en TypeScript con Vite. La simulación y MLflow están en Render, con Docker y
> HTTPS. Cada visitante juega su propia partida.
> Además tenemos monitoreo: éxito y causas de fallo por nivel, latencia y versión servida. El rollback lo
> probamos con un simulacro real.

## 10 · Demo (los dos, 2 min)

Abran https://parking-rl.vercel.app unos minutos antes, para que el servidor no esté dormido.

1. **Nivel 5** (Esteban): *«Aquí el auto entra por abajo y tiene que rodear las filas para encontrar el cupo
   libre.»* Dejarlo estacionar una vez.
2. **Selector de nivel → nivel 3** (Lizeth): *«Cuando una persona cruza, el auto frena y espera.»*
3. **Botón Métricas** (Lizeth): *«Aquí está el éxito en la prueba fija y en el examen final, la comparación con
   la regla y la identidad del modelo: su commit y su huella SHA-256.»*
4. **Panel → Evaluar y comparar** (Esteban): elegir la regla y el modelo y pulsar Evaluar. *«Los mismos casos
   para los dos.»*
5. **«Ver este run en MLflow»** (Esteban): mostrar el run y volver.
6. **«Volver a la versión anterior»** (Lizeth): *«Así se hace un rollback y se ve qué modelo maneja.»*

**Si algo falla:** abrir el plan B en el computador (`.venv/bin/python run.py serve --level N5 --port 8765`,
luego http://127.0.0.1:8765).

## 11 · Limitaciones y aprendizajes (Esteban, 30 s)

> **Limitaciones.**
> - La física es simplificada.
> - El freno de emergencia conoce la posición real del peatón; un auto real dependería de sensores con error.
> - El servidor gratuito se duerme y tiene poco procesador.
>
> **Lo que nos llevamos.**
> - Mirar la escena 3D encontró errores que las métricas no mostraban.
> - El currículo importó más que los hiperparámetros.
> - La trazabilidad no fue un trámite: nos ayudó a encontrar y corregir errores reales.

## 12 · Cierre (los dos, 10 s)

> Gracias. Los enlaces al juego, al repositorio y a MLflow están en pantalla. ¿Preguntas?

---

## Preguntas probables (para prepararse)

- **¿Por qué PPO y no DQN?** DQN no superó a la regla en N1 (42 / 0 / 8 % según la semilla); PPO con currículo
  sí, y es la referencia del curso.
- **¿Cómo saben que el resultado no es suerte?** Hay 3 semillas por receta, una prueba fija de 100 casos y un
  examen final con semillas secretas que se usa una sola vez.
- **¿Qué pasa si cambian el ambiente?** Sube la versión. Los modelos viejos quedan como «otra versión del
  nivel» y no se mezclan; para la versión nueva se crea un examen final nuevo.
- **¿Cómo vuelven a un modelo anterior?** Con `python -m agents.serving rollback NX`, que queda en git; Render
  vuelve a publicar. En la web también se puede probar en la propia partida.
- **¿El freno de emergencia no es hacer trampa?** Es una capa de seguridad documentada, como el AEB de los autos
  reales. Forma parte de la versión del ambiente, y se entrena y evalúa con ella.
