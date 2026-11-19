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
# v11: zona de precaución — con un peatón cerca adelante, no se pasa de esta velocidad
CAUTION_SPEED = 1.2      # m/s (~4 km/h)
CAUTION_AHEAD = 4.0      # m delante del parachoques
CAUTION_SIDE = 2.5       # m a cada lado del eje del auto


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


def pedestrian_near_ahead(env) -> bool:
    """v11: ¿hay un peatón cerca delante (o detrás si va en reversa)?"""
    cfg = env.cfg
    direction = -1.0 if env.speed < -0.05 else 1.0
    c, s = np.cos(env.heading), np.sin(env.heading)
    for p in env.pedestrians:
        px, py = p["x"] - env.pos[0], p["y"] - env.pos[1]
        ahead = (px * c + py * s) * direction
        side = -px * s + py * c
        if 0.0 < ahead < cfg.car_length / 2 + CAUTION_AHEAD and abs(side) < CAUTION_SIDE:
            return True
    return False


def apply(env, action: int, caution: bool = False) -> tuple[int, bool]:
    """Acción final y si el freno de emergencia intervino.

    caution (v11): además del freno, con un peatón cerca adelante no deja pasar de CAUTION_SPEED.
    """
    if action == BRAKE:
        return action, False
    if pedestrian_in_path(env, action):
        return BRAKE, True
    if caution and abs(env.speed) > CAUTION_SPEED and action in (FORWARD, REVERSE) and pedestrian_near_ahead(env):
        return BRAKE, True
    return action, False

