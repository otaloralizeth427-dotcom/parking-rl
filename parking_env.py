"""Ambiente Gymnasium de estacionamiento autónomo.

Coordenadas del mundo en metros: x a lo largo del pasillo, y perpendicular.
El vehículo sigue un modelo cinemático de bicicleta.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Acciones discretas
FORWARD, REVERSE, LEFT, RIGHT, BRAKE = range(5)
ACTION_NAMES = ["Avanzar", "Retroceder", "Izquierda", "Derecha", "Frenar"]

DIFFICULTIES = ("easy", "normal", "hard")


@dataclass(frozen=True)
class ParkingConfig:
    lot_width: float = 30.0          # extensión en x
    lot_height: float = 20.0         # extensión en y
    slot_width: float = 3.0
    slot_depth: float = 5.5
    slots_per_row: int = 8
    car_length: float = 4.2
    car_width: float = 1.8
    wheelbase: float = 2.6
    max_steer: float = 0.6           # rad
    steer_step: float = 0.15         # rad por acción
    accel: float = 2.0               # m/s²
    brake_decel: float = 5.0         # m/s²
    max_speed: float = 3.0
    max_reverse_speed: float = 2.0
    drag: float = 0.97               # factor de rodadura por paso
    dt: float = 0.2
    substeps: int = 2
    max_steps: int = 200
    n_sensors: int = 8
    sensor_range: float = 8.0
    occupancy: float = 0.6           # probabilidad de que un cupo vecino esté ocupado
    park_heading_tol: float = 0.25   # rad
    park_speed_tol: float = 0.4
    park_margin: float = 0.15        # holgura de las esquinas fuera del cupo
    # Recompensas
    r_time: float = -0.05
    r_idle: float = -0.15            # detenido fuera del cupo: evita políticas que se "congelan"
    idle_speed: float = 0.1
    r_collision: float = -50.0
    r_success: float = 100.0
    w_distance: float = 1.0
    w_heading: float = 1.5
    w_slot_speed: float = 0.3


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fold_heading(a: float) -> float:
    """Error de orientación ignorando el sentido (se acepta entrar de frente o en reversa)."""
    a = wrap_angle(a)
    if a > np.pi / 2:
        a -= np.pi
    elif a < -np.pi / 2:
        a += np.pi
    return float(a)


def rot(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def box_corners(center, half, angle) -> np.ndarray:
    local = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * half
    return local @ rot(angle).T + center


def obb_overlaps(center, half, angle, centers, halves, angles) -> np.ndarray:
    """Teorema del eje separador: una caja contra N cajas orientadas (vectorizado)."""
    ca, sa = np.cos(angle), np.sin(angle)
    a_axes = np.array([[ca, sa], [-sa, ca]])                      # (2,2)
    cb, sb = np.cos(angles), np.sin(angles)
    b_axes = np.stack([np.stack([cb, sb], -1), np.stack([-sb, cb], -1)], 1)  # (N,2,2)
    d = centers - center                                            # (N,2)
    overlap = np.ones(len(centers), dtype=bool)
    axes = [np.broadcast_to(a_axes[0], d.shape), np.broadcast_to(a_axes[1], d.shape),
            b_axes[:, 0], b_axes[:, 1]]
    for n in axes:
        ra = half[0] * np.abs(n @ a_axes[0]) + half[1] * np.abs(n @ a_axes[1])
        rb = halves[:, 0] * np.abs(np.sum(n * b_axes[:, 0], -1)) + \
            halves[:, 1] * np.abs(np.sum(n * b_axes[:, 1], -1))
        overlap &= np.abs(np.sum(d * n, -1)) <= ra + rb
    return overlap


def raycast_obbs(origin, directions, centers, halves, angles, max_range) -> np.ndarray:
    """Distancia del primer impacto de cada rayo contra N cajas (método de slabs)."""
    c, s = np.cos(angles), np.sin(angles)
    d = origin - centers                                            # (N,2)
    o_local = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)
    dir_x = directions[:, None, 0] * c + directions[:, None, 1] * s  # (R,N)
    dir_y = -directions[:, None, 0] * s + directions[:, None, 1] * c
    dirs = np.stack([dir_x, dir_y], -1)                              # (R,N,2)
    dirs = np.where(np.abs(dirs) < 1e-9, 1e-9, dirs)
    t1 = (-halves - o_local) / dirs
    t2 = (halves - o_local) / dirs
    t_near = np.minimum(t1, t2).max(-1)
    t_far = np.maximum(t1, t2).min(-1)
    hit = (t_far >= np.maximum(t_near, 0.0))
    t = np.where(t_near > 0, t_near, 0.0)
    t = np.where(hit, t, max_range)
    return np.minimum(t.min(-1), max_range)


class ParkingEnv(gym.Env):
    """El vehículo debe estacionarse en el cupo objetivo sin chocar.

    Observación (float32, 14 valores):
        0-1  posición del cupo en el marco del vehículo (dx adelante, dy izquierda) / 10
        2    error de orientación respecto al cupo (plegado a ±π/2) / (π/2)
        3    velocidad / velocidad máxima
        4    ángulo de dirección / dirección máxima
        5    1 si el vehículo está dentro del cupo
        6-13 distancias de los sensores (8 rayos) / alcance
    """

    metadata = {"render_modes": []}

    def __init__(self, difficulty: str = "normal", sensor_noise: float = 0.0,
                 config: ParkingConfig | None = None):
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty debe ser uno de {DIFFICULTIES}, no {difficulty!r}")
        if sensor_noise < 0:
            raise ValueError("sensor_noise debe ser >= 0")
        self.difficulty = difficulty
        self.sensor_noise = sensor_noise
        self.cfg = config or ParkingConfig()

        n_obs = 6 + self.cfg.n_sensors
        low = np.full(n_obs, -1.0, dtype=np.float32)
        high = np.full(n_obs, 1.0, dtype=np.float32)
        low[:2], high[:2] = -5.0, 5.0
        low[5:] = 0.0
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        # Rayos relativos al vehículo: frente, diagonales, lados, traseras, atrás
        self.sensor_angles = np.linspace(0, 2 * np.pi, self.cfg.n_sensors, endpoint=False)
        self._car_half = np.array([self.cfg.car_length / 2, self.cfg.car_width / 2])
        self._walls = self._build_walls()
        self.slots = self._build_slots()
        self._state_ready = False

    # ------------------------------------------------------------------ layout
    def _build_slots(self) -> list[dict]:
        cfg = self.cfg
        slots = []
        x0 = -(cfg.slots_per_row - 1) * cfg.slot_width / 2
        y_center = cfg.lot_height / 2 - cfg.slot_depth / 2
        for row, sign in enumerate((1, -1)):
            for i in range(cfg.slots_per_row):
                slots.append({
                    "id": row * cfg.slots_per_row + i,
                    "center": np.array([x0 + i * cfg.slot_width, sign * y_center]),
                    # El eje largo del cupo es vertical (y)
                    "angle": np.pi / 2,
                    "row": row,
                })
        return slots

    def _build_walls(self) -> list[dict]:
        w, h, t = self.cfg.lot_width / 2, self.cfg.lot_height / 2, 0.3
        return [
            {"center": (0, h + t), "half": (w + 2 * t, t), "kind": "curb"},
            {"center": (0, -h - t), "half": (w + 2 * t, t), "kind": "curb"},
            {"center": (w + t, 0), "half": (t, h + 2 * t), "kind": "curb"},
            {"center": (-w - t, 0), "half": (t, h + 2 * t), "kind": "curb"},
        ]

    def _sample_layout(self):
        rng, cfg = self.np_random, self.cfg
        self.target = self.slots[int(rng.integers(len(self.slots)))]
        tc = self.target["center"]
        row_sign = 1 if self.target["row"] == 0 else -1

        obstacles = [dict(o, angle=0.0) for o in self._walls]
        for slot in self.slots:
            if slot is self.target:
                continue
            # Los vecinos directos se ocupan con más frecuencia para exigir precisión
            neighbour = slot["row"] == self.target["row"] and \
                abs(slot["center"][0] - tc[0]) < cfg.slot_width * 1.5
            p = 0.85 if neighbour else cfg.occupancy
            if rng.random() < p:
                jitter = rng.uniform(-0.15, 0.15, size=2)
                obstacles.append({
                    "center": tuple(slot["center"] + jitter),
                    "half": (cfg.car_width / 2, cfg.car_length / 2),
                    "angle": float(rng.uniform(-0.05, 0.05)),
                    "kind": "car",
                    "color": int(rng.integers(6)),
                })
        # Columnas al final de cada fila
        for sx in (-1, 1):
            for sy in (-1, 1):
                x = sx * (cfg.slots_per_row * cfg.slot_width / 2 + 1.0)
                obstacles.append({"center": (x, sy * (cfg.lot_height / 2 - 1.2)),
                                  "half": (0.4, 0.4), "angle": 0.0, "kind": "pillar"})

        # Pose inicial en el pasillo
        aisle = cfg.lot_height / 2 - cfg.slot_depth
        if self.difficulty == "easy":
            x = tc[0] + rng.uniform(-1.0, 1.0)
            y = row_sign * rng.uniform(-1.5, 1.0)
            heading = row_sign * np.pi / 2 + rng.uniform(-0.3, 0.3)
        else:
            side = rng.choice([-1, 1])
            limit = cfg.lot_width / 2 - 3.0
            x = float(np.clip(tc[0] + side * rng.uniform(4.0, 8.0), -limit, limit))
            if abs(x - tc[0]) < 3.0:  # rebotó contra el borde: usar el otro lado
                x = float(np.clip(tc[0] - side * rng.uniform(4.0, 8.0), -limit, limit))
            y = rng.uniform(-1.5, 1.5)
            heading = (np.pi if x > tc[0] else 0.0) + rng.uniform(-0.2, 0.2)

        n_cones = {"easy": 0, "normal": 2, "hard": 5}[self.difficulty]
        placed = 0
        attempts = 0
        while placed < n_cones and attempts < 200:
            attempts += 1
            p = np.array([rng.uniform(-cfg.lot_width / 2 + 1, cfg.lot_width / 2 - 1),
                          rng.uniform(-aisle + 0.8, aisle - 0.8)])
            if np.linalg.norm(p - (x, y)) < 3.5:
                continue
            if abs(p[0] - tc[0]) < 2.5:          # deja libre la entrada al cupo
                continue
            on_path = min(x, tc[0]) - 1 < p[0] < max(x, tc[0]) + 1 and abs(p[1] - y) < 2.2
            if self.difficulty == "normal" and on_path:
                continue
            obstacles.append({"center": tuple(p), "half": (0.25, 0.25), "angle": 0.0, "kind": "cone"})
            placed += 1

        self.obstacles = obstacles
        self._obs_centers = np.array([o["center"] for o in obstacles], dtype=float)
        self._obs_halves = np.array([o["half"] for o in obstacles], dtype=float)
        self._obs_angles = np.array([o["angle"] for o in obstacles], dtype=float)
        return np.array([x, y], dtype=float), float(wrap_angle(heading))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.pos, self.heading = self._sample_layout()
        self.speed = 0.0
        self.steer = 0.0
        self.steps = 0
        self.trajectory = [self.pos.copy()]
        self._state_ready = True
        self._sensors = self._read_sensors()
        self._prev_potential = self._potential()
        return self._observation(), self._info(collision=False, success=False)

    def step(self, action):
        if not self._state_ready:
            raise RuntimeError("Llama a reset() antes de step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action!r}")
        cfg = self.cfg
        action = int(action)

        if action == FORWARD:
            self.speed += cfg.accel * cfg.dt
        elif action == REVERSE:
            self.speed -= cfg.accel * cfg.dt
        elif action == LEFT:
            self.steer += cfg.steer_step
        elif action == RIGHT:
            self.steer -= cfg.steer_step
        elif action == BRAKE:
            dv = cfg.brake_decel * cfg.dt
            self.speed = 0.0 if abs(self.speed) <= dv else self.speed - np.sign(self.speed) * dv
        self.speed = float(np.clip(self.speed * cfg.drag, -cfg.max_reverse_speed, cfg.max_speed))
        self.steer = float(np.clip(self.steer, -cfg.max_steer, cfg.max_steer))

        collision = False
        h = cfg.dt / cfg.substeps
        for _ in range(cfg.substeps):
            self.pos = self.pos + self.speed * h * np.array([np.cos(self.heading), np.sin(self.heading)])
            self.heading = float(wrap_angle(self.heading + self.speed / cfg.wheelbase * np.tan(self.steer) * h))
            if self._collides():
                collision = True
                break
        self.steps += 1
        self.trajectory.append(self.pos.copy())
        self._sensors = self._read_sensors()

        success = not collision and self._is_parked()
        potential = self._potential()
        reward = cfg.r_time + potential - self._prev_potential
        self._prev_potential = potential
        if self._corners_in_slot():
            reward -= cfg.w_slot_speed * abs(self.speed)
        elif abs(self.speed) < cfg.idle_speed:
            reward += cfg.r_idle
        if collision:
            reward += cfg.r_collision
        if success:
            reward += cfg.r_success

        terminated = collision or success
        truncated = not terminated and self.steps >= cfg.max_steps
        return self._observation(), float(reward), terminated, truncated, \
            self._info(collision=collision, success=success)

    # ------------------------------------------------------------------ helpers
    def _relative_target(self):
        delta = self.target["center"] - self.pos
        c, s = np.cos(self.heading), np.sin(self.heading)
        return np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])

    def heading_error(self) -> float:
        return fold_heading(self.heading - self.target["angle"])

    def distance(self) -> float:
        return float(np.linalg.norm(self.target["center"] - self.pos))

    def _potential(self) -> float:
        # La alineación pesa más cuanto más cerca está el vehículo del cupo
        dist = self.distance()
        closeness = np.clip(1.0 - dist / 6.0, 0.0, 1.0)
        return -self.cfg.w_distance * dist - self.cfg.w_heading * closeness * abs(self.heading_error())

    def _collides(self) -> bool:
        w, h = self.cfg.lot_width / 2, self.cfg.lot_height / 2
        corners = box_corners(self.pos, self._car_half, self.heading)
        if np.any(np.abs(corners[:, 0]) > w) or np.any(np.abs(corners[:, 1]) > h):
            return True
        return bool(obb_overlaps(self.pos, self._car_half, self.heading,
                                 self._obs_centers, self._obs_halves, self._obs_angles).any())

    def _corners_in_slot(self, margin: float = 0.0) -> bool:
        slot = self.target
        corners = box_corners(self.pos, self._car_half, self.heading) - slot["center"]
        local = corners @ rot(slot["angle"])       # proyección en los ejes del cupo
        half = np.array([self.cfg.slot_depth / 2, self.cfg.slot_width / 2]) + margin
        return bool(np.all(np.abs(local) <= half))

    def _is_parked(self) -> bool:
        cfg = self.cfg
        return (self._corners_in_slot(cfg.park_margin)
                and abs(self.heading_error()) <= cfg.park_heading_tol
                and abs(self.speed) <= cfg.park_speed_tol)

    def _read_sensors(self) -> np.ndarray:
        angles = self.sensor_angles
        dirs = np.stack([np.cos(self.heading + angles), np.sin(self.heading + angles)], -1)
        hits = raycast_obbs(self.pos, dirs, self._obs_centers, self._obs_halves,
                            self._obs_angles, self.cfg.sensor_range + 3.0)
        # Distancia medida desde la carrocería, no desde el centro
        l, w = self._car_half
        with np.errstate(divide="ignore"):
            body = np.minimum(l / np.abs(np.cos(angles)), w / np.abs(np.sin(angles)))
        dist = np.clip(hits - body, 0.0, self.cfg.sensor_range)
        if self.sensor_noise > 0:
            dist = np.clip(dist + self.np_random.normal(0, self.sensor_noise, dist.shape),
                           0.0, self.cfg.sensor_range)
        return dist

    def _observation(self) -> np.ndarray:
        cfg = self.cfg
        rel = self._relative_target()
        if self.sensor_noise > 0:
            rel = rel + self.np_random.normal(0, self.sensor_noise * 0.5, 2)
        obs = np.concatenate([
            np.clip(rel / 10.0, -5, 5),
            [self.heading_error() / (np.pi / 2),
             np.clip(self.speed / cfg.max_speed, -1, 1),
             self.steer / cfg.max_steer,
             float(self._corners_in_slot())],
            self._sensors / cfg.sensor_range,
        ])
        return obs.astype(np.float32)

    def _info(self, collision: bool, success: bool) -> dict:
        return {
            "distance": self.distance(),
            "heading_error": self.heading_error(),
            "speed": self.speed,
            "steer": self.steer,
            "collision": collision,
            "success": success,
            "in_slot": self._corners_in_slot(),
            "steps": self.steps,
            "target_id": self.target["id"],
        }

    def render_state(self) -> dict:
        """Estado serializable para la visualización 3D."""
        if not self._state_ready:
            raise RuntimeError("Llama a reset() antes de render_state()")
        cfg = self.cfg
        return {
            "car": {"x": float(self.pos[0]), "y": float(self.pos[1]), "heading": self.heading,
                    "speed": self.speed, "steer": self.steer,
                    "length": cfg.car_length, "width": cfg.car_width},
            "target": {"id": self.target["id"], "x": float(self.target["center"][0]),
                       "y": float(self.target["center"][1]), "angle": self.target["angle"]},
            "sensors": [{"angle": float(a), "distance": float(d)}
                        for a, d in zip(self.sensor_angles, self._sensors)],
            "sensor_range": cfg.sensor_range,
            "distance": self.distance(),
            "heading_error": self.heading_error(),
            "in_slot": self._corners_in_slot(),
            "steps": self.steps,
        }

    def layout(self) -> dict:
        """Geometría estática del episodio (cupos, obstáculos, dimensiones)."""
        cfg = self.cfg
        return {
            "lot": {"width": cfg.lot_width, "height": cfg.lot_height},
            "slots": [{"id": s["id"], "x": float(s["center"][0]), "y": float(s["center"][1]),
                       "width": cfg.slot_width, "depth": cfg.slot_depth, "row": s["row"]}
                      for s in self.slots],
            "obstacles": [{"kind": o["kind"], "x": float(o["center"][0]), "y": float(o["center"][1]),
                           "hx": float(o["half"][0]), "hy": float(o["half"][1]),
                           "angle": float(o["angle"]), "color": o.get("color", 0)}
                          for o in self.obstacles],
        }
