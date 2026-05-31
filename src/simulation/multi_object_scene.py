"""Helpers para escenas multi-object (Iter 4): spawn, paint, sample posiciones."""
from __future__ import annotations

import numpy as np

BIN_X_RANGE = (0.38, 0.55)
BIN_Y_RANGE = (-0.17, -0.02)
Z_FIXED = 0.033
MIN_DIST_M = 0.04
MAX_PREEXISTING = 5  # /object_1 .. /object_5 ya están en la escena
MAX_TOTAL_CUBES = 8  # rango (3, 8) — pre-creamos 8 una vez por bridge
PARK_POSITION = (-1.0, -1.0, -1.0)  # fuera del bin y de la cámara

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


def _list_existing_cubes(sim) -> list[int]:
    """Devuelve TODOS los handles con alias que empieza con 'object_' (incluye clones)."""
    handles: list[int] = []
    objs = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 1 + 2)
    for h in objs:
        try:
            alias = sim.getObjectAlias(h, 4)
        except Exception:
            continue
        if alias.lower().startswith("object_"):
            handles.append(h)
    # Sort by handle for deterministic order (pre-existing tienen handles más bajos)
    handles.sort()
    return handles


def ensure_n_cubes(sim, n_needed: int) -> list[int]:
    """Devuelve handles para n_needed cubos. Reusa todo lo existente (clones incluidos)."""
    existing = _list_existing_cubes(sim)
    if not existing:
        raise RuntimeError("escena no tiene /object_*; ¿es bin_base.ttt?")
    base = existing[0]
    handles = list(existing)
    while len(handles) < n_needed:
        new = sim.copyPasteObjects([base], 0)
        if not new:
            raise RuntimeError("copyPasteObjects falló")
        handles.append(new[0])
    return handles[:n_needed]


def setup_multi_object_scene(
    sim, n_cubes: int, rng: np.random.Generator
) -> tuple[list[int], np.ndarray]:
    """Spawn + paint n_cubes cubos. Los cubos no usados se mueven a PARK_POSITION.

    handles[0] = target rojo; resto = distractor azul/verde.
    """
    # Asegurar MAX_TOTAL_CUBES en escena para no acumular clones entre llamadas
    all_handles = ensure_n_cubes(sim, MAX_TOTAL_CUBES)
    active = all_handles[:n_cubes]
    inactive = all_handles[n_cubes:]
    for h in inactive:
        sim.setObjectPosition(h, -1, list(PARK_POSITION))
    positions = sample_non_overlapping_positions(n_cubes, rng)
    for h, pos in zip(active, positions):
        sim.setObjectPosition(h, -1, [float(pos[0]), float(pos[1]), float(pos[2])])
    paint_cube(sim, active[0], COLOR_TARGET)
    for h in active[1:]:
        color = COLOR_DISTRACTOR_POOL[int(rng.integers(0, len(COLOR_DISTRACTOR_POOL)))]
        paint_cube(sim, h, color)
    return active, positions


def measure_collision(
    sim,
    distractor_handles: list[int],
    initial_positions: np.ndarray,
    threshold_m: float = 0.05,
) -> tuple[bool, float]:
    """Devuelve (collided, max_displacement) sobre los distractors.

    threshold_m=0.05 (5 cm): captura colisión destructiva e ignora "brush"
    leve. Necesario porque el gripper RG2 abierto mide ~8.5 cm de ancho y
    los cubos pueden estar a 4 cm de distancia mínima.
    """
    if len(distractor_handles) == 0:
        return False, 0.0
    max_disp = 0.0
    DISP_CLIP_M = 1.0  # cubos que vuelan fuera del bin se clampean a 1 m
    for h, pos0 in zip(distractor_handles, initial_positions):
        pos = sim.getObjectPosition(h, -1)
        d = float(np.linalg.norm(np.array(pos) - np.array(pos0)))
        d = min(d, DISP_CLIP_M)
        max_disp = max(max_disp, d)
    return max_disp > threshold_m, max_disp
