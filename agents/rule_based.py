"""Política de referencia por reglas (sin aprendizaje).

Usa solo la observación: persigue un punto sobre el eje del cupo que se acerca al centro a
medida que el vehículo se alinea con ese eje (seguimiento "pure pursuit"), y frena dentro del cupo.
"""

from __future__ import annotations

import numpy as np

from envs.parking_env import BRAKE, FORWARD, LEFT, RIGHT, ParkingConfig

CFG = ParkingConfig()


class RuleBasedPolicy:
    name = "rule"

    def __init__(self, approach_gain: float = 1.2, max_approach: float = 7.0,
                 cruise_speed: float = 1.2, slot_speed: float = 0.6):
        self.approach_gain = approach_gain
        self.max_approach = max_approach
        self.cruise_speed = cruise_speed
        self.slot_speed = slot_speed

    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        target = obs[:2] * 10.0                       # centro del cupo en el marco del vehículo
        heading_err = obs[2] * np.pi / 2
        speed, steer = obs[3] * CFG.max_speed, obs[4] * CFG.max_steer
        in_slot = obs[5] > 0.5
        front_clear = (obs[6:] * CFG.sensor_range)[[7, 0, 1]].min()

        # Eje del cupo en el marco del vehículo, orientado hacia donde mira el vehículo
        axis = np.array([np.cos(heading_err), -np.sin(heading_err)])
        if axis[0] < 0:
            axis = -axis
        along = np.dot(-target, axis)                 # < 0 antes del centro del cupo
        lateral = abs(axis[0] * -target[1] - axis[1] * -target[0])
        # Punto guía sobre el eje, más adelante cuanto más lejos está el vehículo del eje
        lookahead = float(np.clip(self.approach_gain * lateral, 1.5, self.max_approach))
        aim = target + axis * (along + lookahead)

        if in_slot and along > -0.3:
            return BRAKE

        # Pure pursuit: curvatura necesaria para pasar por el punto guía
        distance = max(np.linalg.norm(aim), 1e-3)
        desired_steer = np.clip(np.arctan(2 * CFG.wheelbase * aim[1] / distance ** 2),
                                -CFG.max_steer, CFG.max_steer)
        near = lateral < 1.0 and along > -3.0
        desired_speed = max(0.35, min(self.slot_speed, 0.25 * abs(along))) if near else self.cruise_speed

        if front_clear < 0.4 and speed > 0:
            return BRAKE
        if speed < 0.3:
            return FORWARD
        if abs(desired_steer - steer) > CFG.steer_step / 2:
            return LEFT if desired_steer > steer else RIGHT
        if speed > desired_speed + 0.4:
            return BRAKE
        return FORWARD if speed < desired_speed else (LEFT if desired_steer > steer else RIGHT)
