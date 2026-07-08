#!/usr/bin/env python3
"""Collector Iter 9 — clutter randomization: distractores en las capturas.

Igual que v8 (1000 trayectorias heurísticas, apariencia del objetivo
randomizada en forma+color), pero cada captura RGB-D incluye además 0-2 piezas
DISTRACTORAS de forma/color aleatorios, separadas ≥9 cm del objetivo. El plan
heurístico depende solo de la pose del objetivo → las trayectorias no cambian:
el sistema aprende a usar la pose del conditioning para ir por la pieza
indicada e ignorar el resto de la escena (selección en clutter).

Uso (CoppeliaSim en :23000):
    python experiments/collect_diffusion_dataset_v9_clutter.py [--n 1000]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.collect_diffusion_dataset import _capture_rgbd_only, sample_pose
from experiments.collect_diffusion_dataset_v8_randomized import (
    COLORES,
    FORMAS,
    PALETA,
    PARK,
    _setup_piezas,
)
from src.planning.diffusion_policy import DiffusionGraspPlanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("collect_v9clut")

DATASET_DIR = REPO / "data" / "datasets" / "sim_pick_v9clut"
SEED = 42
CHUNK_SIZE = 200

MAX_DISTRACTORES = 2
SEP_MIN_M = 0.09
DIST_X = (0.38, 0.58)
DIST_Y = (-0.19, -0.01)
Z_PIEZA = 0.033

def park_piezas_fijas(sim) -> None:
    """Estaciona object_2..5 (piezas fijas de bin_base.ttt) fuera de escena y
    las hace estáticas. Así los distractores que creamos no chocan con ellas
    (evita expulsiones violentas) y no aparecen en las capturas RGB-D: el
    clutter de la escena es exactamente el que controlamos."""
    for k in range(2, 6):
        try:
            h = sim.getObject(f"/object_{k}")
        except Exception:
            continue
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
        sim.setObjectPosition(h, -1, [-1.5, -1.5 - 0.2 * k, -0.5])


def _setup_distractores(sim) -> list[dict]:
    """Pool de MAX_DISTRACTORES slots × 3 formas, estacionados fuera de cámara."""
    slots = []
    for s in range(MAX_DISTRACTORES):
        slot = {}
        for forma, tipo_attr, pure_code in (("cubo", "primitiveshape_cuboid", 0),
                                            ("esfera", "primitiveshape_spheroid", 1),
                                            ("cilindro", "primitiveshape_cylinder", 2)):
            try:
                h = sim.createPrimitiveShape(getattr(sim, tipo_attr), [0.05, 0.05, 0.05], 0)
            except Exception:
                h = sim.createPureShape(pure_code, 8, [0.05, 0.05, 0.05], 0.05)
            sim.setObjectPosition(h, -1, [PARK[0] - 1.0 - 0.2 * len(slot), PARK[1] - 0.3 * s, PARK[2]])
            slot[forma] = h
        slots.append(slot)
    return slots


def _muestrear_distraccion(rng, target_xy) -> list[tuple]:
    """(forma_idx, color_idx, x, y) por distractor. Consume SIEMPRE el mismo
    número de draws del rng → resume determinista por replay."""
    n = int(rng.integers(0, MAX_DISTRACTORES + 1))
    out = []
    puestos = [tuple(target_xy)]
    for _ in range(MAX_DISTRACTORES):
        fi = int(rng.integers(0, 3))
        ci = int(rng.integers(0, len(COLORES)))
        # posición por rejection sampling con draws acotados (determinista)
        x = y = None
        for _try in range(50):
            cx = float(rng.uniform(*DIST_X))
            cy = float(rng.uniform(*DIST_Y))
            if all((cx - px) ** 2 + (cy - py) ** 2 >= SEP_MIN_M**2 for px, py in puestos):
                x, y = cx, cy
                break
        if len(out) < n and x is not None:
            puestos.append((x, y))
            out.append((fi, ci, x, y))
    return out


def _capture_clutter(bridge, pose, piezas, slots, fi, ci, distraccion) -> np.ndarray:
    sim = bridge.sim
    # estacionar todo
    for k, (f, h) in enumerate(piezas.items()):
        sim.setObjectPosition(h, -1, [PARK[0] - 0.2 * k, PARK[1], PARK[2]])
    for s, slot in enumerate(slots):
        for k, h in enumerate(slot.values()):
            sim.setObjectPosition(h, -1, [PARK[0] - 1.0 - 0.2 * k, PARK[1] - 0.3 * s, PARK[2]])
    # objetivo
    h = piezas[FORMAS[fi]]
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, list(PALETA[COLORES[ci]]))
    sim.setObjectPosition(h, -1, [float(p) for p in pose[:3, 3]])
    # distractores
    for s, (dfi, dci, x, y) in enumerate(distraccion):
        hd = slots[s][FORMAS[dfi]]
        sim.setShapeColor(hd, None, sim.colorcomponent_ambient_diffuse,
                          list(PALETA[COLORES[dci]]))
        sim.setObjectPosition(hd, -1, [x, y, Z_PIEZA])
    return _capture_rgbd_only(bridge)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    n_total = args.n

    from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out = DATASET_DIR / "heuristic.pt"

    start, poses, rgbds, trajs, formas_idx, colores_idx, n_dist = 0, None, None, None, None, None, None
    if out.exists():
        d = torch.load(out, weights_only=True)
        if d.get("seed") == SEED and len(d["poses"]) == n_total:
            done = int(d.get("n_valid", 0))
            if done >= n_total:
                logger.info(f"Ya hay {done} en {out}; nada que hacer")
                return 0
            logger.info(f"Resume desde {done}/{n_total}")
            poses, rgbds, trajs = d["poses"].numpy(), d["rgbds"].numpy(), d["trajs"].numpy()
            formas_idx, colores_idx = d["formas"].numpy(), d["colores"].numpy()
            n_dist = d["n_distractores"].numpy()
            start = done

    if poses is None:
        poses = np.zeros((n_total, 16), dtype=np.float32)
        rgbds = np.zeros((n_total, 4, 224, 224), dtype=np.float32)
        trajs = np.zeros((n_total, 16, 7), dtype=np.float32)
        formas_idx = np.zeros(n_total, dtype=np.int8)
        colores_idx = np.zeros(n_total, dtype=np.int8)
        n_dist = np.zeros(n_total, dtype=np.int8)

    planner = DiffusionGraspPlanner(action_dim=7, horizon=16, n_diffusion_steps=100, device="cpu")
    rng_pose = np.random.default_rng(SEED)
    rng_app = np.random.default_rng(SEED + 7)
    rng_clut = np.random.default_rng(SEED + 13)
    for _ in range(start):           # replay para resume determinista
        pose = sample_pose(rng_pose)
        rng_app.integers(0, 3)
        rng_app.integers(0, len(COLORES))
        _muestrear_distraccion(rng_clut, pose[:2, 3])

    SCENE = REPO / "data" / "scenes" / "bin_base.ttt"
    i = start
    logger.info(f"Iter 9: {n_total - start} trayectorias con clutter randomizado")

    while i < n_total:
        chunk_end = min(i + CHUNK_SIZE, n_total)
        with CoppeliaSimBridge() as bridge:
            bridge.load_scene(SCENE)
            park_piezas_fijas(bridge.sim)
            piezas = _setup_piezas(bridge.sim)
            slots = _setup_distractores(bridge.sim)
            bridge.set_stepping(True)
            bridge.start_simulation()
            try:
                while i < chunk_end:
                    pose = sample_pose(rng_pose)
                    fi = int(rng_app.integers(0, 3))
                    ci = int(rng_app.integers(0, len(COLORES)))
                    distraccion = _muestrear_distraccion(rng_clut, pose[:2, 3])
                    rgbd = _capture_clutter(bridge, pose, piezas, slots, fi, ci, distraccion)
                    traj = planner.plan_grasp_heuristic(
                        pose, approach_distance=0.15, lift_height=0.10, with_deposit=True)
                    trajs[i], rgbds[i] = traj[0], rgbd
                    poses[i] = pose.flatten().astype(np.float32)
                    formas_idx[i], colores_idx[i] = fi, ci
                    n_dist[i] = len(distraccion)
                    i += 1
                    if i % 50 == 0:
                        logger.info(f"  {i}/{n_total}")
            finally:
                bridge.stop_simulation()

        torch.save({
            "poses": torch.from_numpy(poses), "rgbds": torch.from_numpy(rgbds),
            "trajs": torch.from_numpy(trajs),
            "formas": torch.from_numpy(formas_idx), "colores": torch.from_numpy(colores_idx),
            "n_distractores": torch.from_numpy(n_dist),
            "formas_nombres": list(FORMAS), "colores_nombres": list(COLORES),
            "source": "heuristic_with_deposit_clutter", "seed": SEED, "n_valid": i,
        }, out)
        logger.info(f"  checkpoint: {i}/{n_total} ({out.stat().st_size / 1e6:.0f} MB)")

    # split 90/10 → train.pt / val.pt (mismo formato que v5/v8)
    rng_split = np.random.default_rng(SEED + 2)
    indices = rng_split.permutation(n_total)
    train_n = int(0.9 * n_total)
    for nombre, idx in (("train", indices[:train_n]), ("val", indices[train_n:])):
        torch.save({
            "poses": torch.from_numpy(poses[idx]),
            "rgbds": torch.from_numpy(rgbds[idx]),
            "trajs": torch.from_numpy(trajs[idx]),
            "formas": torch.from_numpy(formas_idx[idx]),
            "colores": torch.from_numpy(colores_idx[idx]),
        }, DATASET_DIR / f"{nombre}.pt")
        logger.info(f"{nombre}.pt: {len(idx)} muestras")
    logger.info("Colección Iter 9 completa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
