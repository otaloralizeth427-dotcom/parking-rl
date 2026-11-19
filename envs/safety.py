"""Freno de emergencia para peatones (parking-v10, opción safety_brake del ambiente).

Como el frenado autónomo de emergencia (AEB) de un auto real: no reemplaza a quien maneja, lo vigila.
En cada paso proyecta el recorrido del auto hasta poder detenerse y el de cada peatón; si un peatón
va a quedar en ese recorrido, la acción se cambia por FRENAR. Con el auto detenido, no lo deja
arrancar hacia un peatón que está cruzando: espera a que termine de pasar.

Es parte de la dinámica del ambiente (lo que pasa después de cada acción), así que aplica igual al
entrenar, al evaluar, en el test final y en la web, y cambia la versión del ambiente.
"""

from __future__ import annotations

import numpy as np

# Mismos índices que envs.parking_env.ACTION_NAMES (no se importa para evitar un ciclo)
FORWARD, REVERSE, BRAKE = 0, 1, 4

HORIZON_S = 1.2          # cuánto se proyecta el movimiento del peatón
REACTION_S = 0.4         # margen antes de que el freno empiece a actuar
SIDE_MARGIN = 0.6        # holgura lateral además del ancho del auto (m)
FRONT_MARGIN = 0.8       # holgura delante (o detrás) del parachoques (m)


def _stopping_distance(speed: float, decel: float) -> float:
    return speed * speed / (2 * decel) + abs(speed) * REACTION_S


def pedestrian_in_path(env, action: int) -> bool:
    """¿Un peatón queda en el recorrido del auto si se ejecuta `action`?"""
    if not getattr(env, "pedestrians", None):
        return False
    cfg = env.cfg
    speed = float(env.speed)
    # Hacia dónde se va a mover el auto: su velocidad actual o, si está casi quieto, la acción pedida
    if speed > 0.05 or (abs(speed) <= 0.05 and action == FORWARD):
        direction = 1.0
    elif speed < -0.05 or (abs(speed) <= 0.05 and action == REVERSE):
        direction = -1.0
    else:
        return False
    reach = cfg.car_length / 2 + FRONT_MARGIN + _stopping_distance(abs(speed), cfg.brake_decel)
    half_width = cfg.car_width / 2 + SIDE_MARGIN
    c, s = np.cos(env.heading), np.sin(env.heading)
    for p in env.pedestrians:
        heading = p.get("heading", np.pi / 2 if p.get("dir", 1) > 0 else -np.pi / 2)
        v = p.get("speed", 0.0) if p.get("moving", True) else 0.0
        for t in np.linspace(0.0, HORIZON_S, 7):
            px = p["x"] + v * np.cos(heading) * t - env.pos[0]
            py = p["y"] + v * np.sin(heading) * t - env.pos[1]
            ahead = (px * c + py * s) * direction           # a lo largo del auto, en el sentido de avance
            side = -px * s + py * c
            if -cfg.car_length / 2 < ahead < reach and abs(side) < half_width:
                return True
    return False


def apply(env, action: int) -> tuple[int, bool]:
    """Acción final y si el freno de emergencia intervino."""
    if action != BRAKE and pedestrian_in_path(env, action):
        return BRAKE, True
    return action, False

