---
title: Parking RL
emoji: 🚗
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Parking RL · servidor de la simulación

API y WebSocket del proyecto [parking-rl](https://github.com/otaloralizeth427-dotcom/parking-rl).
La web 3D está publicada en Vercel y se conecta aquí. Este Space se arma automáticamente desde
GitHub con `deploy/build_space.py`; no se edita a mano.

- `GET /api/health` → commit servido, niveles y partidas activas.
- `GET /api/config?level=N5` → catálogo de modelos del nivel (con hashes y éxito en la suite).
- `WS /ws?level=N5` → una partida propia por conexión.
