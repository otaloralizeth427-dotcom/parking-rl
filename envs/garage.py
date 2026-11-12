"""Mapa «garaje grande» (nivel N5): azotea con 3 filas de cupos, pasillos laterales y entrada abajo.

Coordenadas en metros, como en ParkingEnv: x horizontal, y vertical (la entrada está en y negativa).

    y = 25  ──────────── pared ────────────
            | fila A (8 cupos, abren abajo) |
            |          pasillo 1            |
            ═══ bordillo ═══════════════════   ← la fila B no se puede cruzar por detrás
            | fila B (8 cupos, abren abajo) |
            |          pasillo 2            |
            ═══ bordillo ═══════════════════
            | fila C (8 cupos, abren abajo) |
            |          pasillo 3            |
    y = -25 ────────── entrada (x = 0) ─────

Los pasillos laterales (|x| > 12.3) conectan todos los pasillos: para llegar a la fila A o B hay que
darle la vuelta a las filas. `RouteField` es la distancia por el camino libre hasta el cupo, calculada con
Dijkstra sobre una cuadrícula de 0.5 m. Solo la usa la recompensa (acercarse por el camino correcto suma):
el agente NO la observa; ve su posición y la del cupo en el mapa y tiene que aprender a rodear las filas.
"""

from __future__ import annotations

import heapq

import numpy as np

LOT_WIDTH, LOT_HEIGHT = 40.0, 50.0
SLOTS_PER_ROW = 8
SLOT_WIDTH, SLOT_DEPTH = 3.0, 5.5
AISLE = 8.5          # ancho de los pasillos frente a las filas (con 6.7 m no cabía el giro de 90°)
ROW_HALF_WIDTH = SLOTS_PER_ROW * SLOT_WIDTH / 2          # 12 m: las filas ocupan |x| <= 12
CURB_HALF = (ROW_HALF_WIDTH + 0.3, 0.15)
# (límite superior en y de la fila, nombre). Todas las filas abren hacia abajo (hacia y negativa).
# Cada fila abre a un pasillo de 8.5 m; detrás de B y C hay un bordillo de 0.3 m.
ROWS = ((LOT_HEIGHT / 2, "A"), (10.7, "B"), (-3.6, "C"))
# Bordillos detrás de las filas B y C: impiden cruzar de un pasillo a otro por un cupo vacío
CURBS_Y = (10.7 + CURB_HALF[1], -3.6 + CURB_HALF[1])
ENTRANCE = np.array([0.0, -22.0])
GRID = 0.5           # resolución de la cuadrícula de la ruta (m)
INFLATE = 0.9        # margen alrededor de los obstáculos (medio ancho del auto)


def build_slots() -> list[dict]:
    slots = []
    x0 = -ROW_HALF_WIDTH + SLOT_WIDTH / 2
    for row, (top, name) in enumerate(ROWS):
        for i in range(SLOTS_PER_ROW):
            slots.append({"id": row * SLOTS_PER_ROW + i, "row": row, "name": f"{name}{i + 1}",
                          "center": np.array([x0 + i * SLOT_WIDTH, top - SLOT_DEPTH / 2]),
                          "angle": np.pi / 2, "opening": -1})
    return slots


def build_curbs() -> list[dict]:
    return [{"center": (0.0, y), "half": CURB_HALF, "angle": 0.0, "kind": "curb"} for y in CURBS_Y]


def aisle_y(slot: dict) -> float:
    """Centro del pasillo al que abre el cupo."""
    return float(slot["center"][1] - SLOT_DEPTH / 2 - AISLE / 2)


def aisle_points(rng, n: int = 1) -> np.ndarray:
    """Puntos al azar sobre el eje de algún pasillo (central o lateral), para arranques variados."""
    pts = []
    for _ in range(n):
        if rng.random() < 0.6:
            top = ROWS[int(rng.integers(len(ROWS)))][0]
            pts.append([rng.uniform(-14.0, 14.0), top - SLOT_DEPTH - AISLE / 2 + rng.uniform(-1.2, 1.2)])
        else:
            pts.append([rng.choice([-1.0, 1.0]) * rng.uniform(15.3, 16.7), rng.uniform(-19.0, 15.0)])
    return np.array(pts)


class RouteField:
    """Distancia por el camino libre hasta un cupo (solo para la recompensa). Se calcula una vez por cupo y se reutiliza.

    Las filas cuentan como obstáculo completo (salvo el cupo objetivo), así que la ruta nunca entra en otro
    cupo. Las celdas de las filas cuestan 20 veces más, los márgenes junto a obstáculos 5 veces más y la ruta
    prefiere el centro de los pasillos (holgura >= 3.5 m): el campo queda definido en todo el lote (aunque el auto se meta en un cupo vacío) y la ruta va por el
    centro de los pasillos. Los bordillos y el exterior son infranqueables.
    """

    _cache: dict[int, "RouteField"] = {}

    def __init__(self, target: dict):
        nx, ny = int(LOT_WIDTH / GRID), int(LOT_HEIGHT / GRID)
        xs = -LOT_WIDTH / 2 + GRID * (np.arange(nx) + 0.5)
        ys = -LOT_HEIGHT / 2 + GRID * (np.arange(ny) + 0.5)
        X, Y = np.meshgrid(xs, ys)                       # índice [iy, ix]
        tc = target["center"]

        row = np.zeros_like(X, dtype=bool)
        for top, _ in ROWS:
            row |= (np.abs(X) <= ROW_HALF_WIDTH) & (Y <= top) & (Y >= top - SLOT_DEPTH)
        free_slot = (np.abs(X - tc[0]) <= SLOT_WIDTH / 2) & (np.abs(Y - tc[1]) <= SLOT_DEPTH / 2)
        row &= ~free_slot
        hard = np.zeros_like(row)
        for y in CURBS_Y:
            hard |= (np.abs(X) <= CURB_HALF[0]) & (np.abs(Y - y) <= max(CURB_HALF[1], GRID / 2 + 0.01))

        def near(mask, margin):
            out = mask.copy()
            k = int(np.ceil(margin / GRID))
            for dy in range(-k, k + 1):
                for dx in range(-k, k + 1):
                    if (dx * dx + dy * dy) * GRID * GRID <= margin * margin:
                        out |= np.roll(np.roll(mask, dy, 0), dx, 1)
            return out

        from scipy.ndimage import distance_transform_edt

        # Holgura de cada celda hasta el obstáculo o el borde más cercano: la ruta prefiere el centro de los
        # pasillos (holgura >= 3.5 m), así el auto tiene espacio para girar en las esquinas de las filas.
        solid = row | hard
        solid[[0, -1], :] = True
        solid[:, [0, -1]] = True
        clearance = distance_transform_edt(~solid) * GRID
        cost = 1.0 + 3.0 * np.clip(1.0 - clearance / 3.5, 0.0, 1.0)
        edge = (np.abs(X) > LOT_WIDTH / 2 - INFLATE) | (np.abs(Y) > LOT_HEIGHT / 2 - INFLATE)
        cost[near(row | hard, INFLATE) | edge] = 5.0
        cost[row] = 20.0
        cost[hard] = np.inf

        dist = np.full(X.shape, np.inf)
        goal = (int((tc[1] + LOT_HEIGHT / 2) / GRID), int((tc[0] + LOT_WIDTH / 2) / GRID))
        dist[goal] = 0.0
        heap = [(0.0, goal)]
        moves = [(dy, dx, GRID * np.hypot(dy, dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
        while heap:
            d, (iy, ix) = heapq.heappop(heap)
            if d > dist[iy, ix]:
                continue
            for dy, dx, step in moves:
                jy, jx = iy + dy, ix + dx
                if 0 <= jy < ny and 0 <= jx < nx and np.isfinite(cost[jy, jx]):
                    nd = d + step * 0.5 * (cost[iy, ix] + cost[jy, jx])
                    if nd < dist[jy, jx]:
                        dist[jy, jx] = nd
                        heapq.heappush(heap, (nd, (jy, jx)))
        # Las celdas de bordillo (infranqueables) toman el valor de su mejor vecina, para que el campo
        # esté definido si el auto queda pegado a uno.
        for iy, ix in zip(*np.where(~np.isfinite(dist))):
            vals = dist[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2]
            dist[iy, ix] = (vals[np.isfinite(vals)].min() + GRID) if np.isfinite(vals).any() else 200.0
        self.dist = dist
        self.target = np.array(tc, dtype=float)

    @classmethod
    def for_slot(cls, target: dict) -> "RouteField":
        if target["id"] not in cls._cache:
            cls._cache[target["id"]] = cls(target)
        return cls._cache[target["id"]]

    def _cell(self, p):
        fx = (p[0] + LOT_WIDTH / 2) / GRID - 0.5
        fy = (p[1] + LOT_HEIGHT / 2) / GRID - 0.5
        ny, nx = self.dist.shape
        return float(np.clip(fx, 0, nx - 1)), float(np.clip(fy, 0, ny - 1))

    def distance(self, p) -> float:
        """Distancia por la ruta (interpolación bilineal). Cerca del cupo coincide con la recta."""
        straight = float(np.linalg.norm(self.target - p))
        if straight < 2.0:
            return straight
        fx, fy = self._cell(p)
        x0, y0 = int(fx), int(fy)
        x1, y1 = min(x0 + 1, self.dist.shape[1] - 1), min(y0 + 1, self.dist.shape[0] - 1)
        tx, ty = fx - x0, fy - y0
        d = self.dist
        v = (d[y0, x0] * (1 - tx) * (1 - ty) + d[y0, x1] * tx * (1 - ty)
             + d[y1, x0] * (1 - tx) * ty + d[y1, x1] * tx * ty)
        routed = float(max(v, straight))
        w = min((straight - 2.0) / 2.0, 1.0)             # transición suave a la recta entre 4 y 2 m del cupo
        return (1 - w) * straight + w * routed

    def waypoint(self, p, lookahead: float = 4.0) -> np.ndarray:
        """Punto de la ruta unos metros más adelante: se baja por el campo de distancias celda a celda."""
        fx, fy = self._cell(p)
        iy, ix = int(round(fy)), int(round(fx))
        ny, nx = self.dist.shape
        for _ in range(int(lookahead / GRID)):
            if self.dist[iy, ix] <= 0.0:
                break
            best = (self.dist[iy, ix], iy, ix)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    jy, jx = iy + dy, ix + dx
                    if 0 <= jy < ny and 0 <= jx < nx and self.dist[jy, jx] < best[0]:
                        best = (self.dist[jy, jx], jy, jx)
            if best[1:] == (iy, ix):
                break
            iy, ix = best[1], best[2]
        point = np.array([-LOT_WIDTH / 2 + GRID * (ix + 0.5), -LOT_HEIGHT / 2 + GRID * (iy + 0.5)])
        # Al final de la ruta el punto es el propio cupo
        if self.distance(p) <= lookahead:
            return self.target.copy()
        return point

    def entrance_path(self) -> tuple[np.ndarray, np.ndarray]:
        """Ruta desde la entrada hasta el cupo (puntos cada 0.5 m) y lo que falta recorrer desde cada punto."""
        if not hasattr(self, "_path"):
            p, points = ENTRANCE.copy(), [ENTRANCE.copy()]
            for _ in range(400):
                if np.linalg.norm(self.target - p) <= GRID:
                    break
                p = self.waypoint(p, GRID)
                points.append(p.copy())
            points.append(self.target.copy())
            pts = np.array(points)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            remaining = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
            self._path = (pts, remaining)
        return self._path

    def start_on_path(self, rng, lo: float, hi: float) -> tuple[np.ndarray, float]:
        """Arranque sobre la ruta de la entrada, a entre `lo` y `hi` metros del cupo (currículo inverso)."""
        pts, remaining = self.entrance_path()
        r = rng.uniform(min(lo, remaining[0]), min(hi, remaining[0]))
        i = int(np.clip(np.searchsorted(-remaining, -r), 1, len(pts) - 1))
        point = pts[i]
        if remaining[i] < SLOT_DEPTH + 1.0:                         # tramo final: derecho hacia el cupo
            return point.astype(float), float(np.pi / 2)
        ahead = pts[min(i + 4, len(pts) - 1)] - pts[i - 1]          # hacia dónde sigue la ruta
        return point.astype(float), float(np.arctan2(ahead[1], ahead[0]))
