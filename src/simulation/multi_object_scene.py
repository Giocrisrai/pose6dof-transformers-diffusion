"""Helpers para escenas multi-object (Iter 4): spawn, paint, sample posiciones."""
from __future__ import annotations

import numpy as np

BIN_X_RANGE = (0.38, 0.55)
BIN_Y_RANGE = (-0.17, -0.02)
Z_FIXED = 0.033
MIN_DIST_M = 0.04
MAX_PREEXISTING = 5  # /object_1 .. /object_5 ya están en la escena

COLOR_TARGET = (0.85, 0.15, 0.15)
COLOR_DISTRACTOR_POOL = [
    (0.15, 0.30, 0.85),
    (0.20, 0.75, 0.20),
]


def sample_non_overlapping_positions(
    n: int, rng: np.random.Generator, max_retries: int = 50
) -> np.ndarray:
    """Sample n posiciones (x, y, z) con distancia mínima MIN_DIST_M entre centros."""
    out: list[np.ndarray] = []
    for _ in range(n):
        for _ in range(max_retries):
            x = rng.uniform(*BIN_X_RANGE)
            y = rng.uniform(*BIN_Y_RANGE)
            p = np.array([x, y, Z_FIXED])
            if all(np.linalg.norm(p[:2] - q[:2]) >= MIN_DIST_M for q in out):
                out.append(p)
                break
        else:
            raise RuntimeError(f"sample failed after {max_retries} retries (n={n})")
    return np.array(out, dtype=np.float32)


def paint_cube(sim, handle: int, color: tuple) -> None:
    sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, list(color))


def ensure_n_cubes(sim, n_needed: int) -> list[int]:
    """Devuelve handles para n_needed cubos; clona /object_1 si hace falta más allá de los 5 pre-existentes."""
    pre_existing: list[int] = []
    for k in range(1, MAX_PREEXISTING + 1):
        try:
            h = sim.getObject(f"/object_{k}")
            pre_existing.append(h)
        except Exception:
            break
    if not pre_existing:
        raise RuntimeError("escena no tiene /object_1; ¿es bin_base.ttt?")
    handles = list(pre_existing[:n_needed])
    base = pre_existing[0]
    while len(handles) < n_needed:
        new = sim.copyPasteObjects([base], 0)
        if not new:
            raise RuntimeError("copyPasteObjects falló")
        handles.append(new[0])
    return handles


def setup_multi_object_scene(
    sim, n_cubes: int, rng: np.random.Generator
) -> tuple[list[int], np.ndarray]:
    """Spawn + paint n_cubes cubos. handles[0] = target rojo; resto = distractor azul/verde."""
    handles = ensure_n_cubes(sim, n_cubes)
    positions = sample_non_overlapping_positions(n_cubes, rng)
    for h, pos in zip(handles, positions):
        sim.setObjectPosition(h, -1, [float(pos[0]), float(pos[1]), float(pos[2])])
    paint_cube(sim, handles[0], COLOR_TARGET)
    for h in handles[1:]:
        color = COLOR_DISTRACTOR_POOL[int(rng.integers(0, len(COLOR_DISTRACTOR_POOL)))]
        paint_cube(sim, h, color)
    return handles, positions


def measure_collision(
    sim,
    distractor_handles: list[int],
    initial_positions: np.ndarray,
    threshold_m: float = 0.01,
) -> tuple[bool, float]:
    """Devuelve (collided, max_displacement) sobre los distractors."""
    if len(distractor_handles) == 0:
        return False, 0.0
    max_disp = 0.0
    for h, pos0 in zip(distractor_handles, initial_positions):
        pos = sim.getObjectPosition(h, -1)
        d = float(np.linalg.norm(np.array(pos) - np.array(pos0)))
        max_disp = max(max_disp, d)
    return max_disp > threshold_m, max_disp
