#!/usr/bin/env python3
"""Collector Iter 8 — domain randomization de apariencia (forma + color).

Igual que v5 (1000 trayectorias heurísticas single-object con deposit), pero la
pieza visible en cada captura RGB-D se randomiza en FORMA (cubo/esfera/cilindro)
y COLOR (paleta de 6). El plan heurístico depende solo de la pose, así que las
trayectorias objetivo no cambian: lo que aprende el sistema es que la apariencia
no debe importar (invariancia del conditioning visual).

Motivación (medido en demo_charla, 2026-06-10): la política v7a entrenada solo
con cubos rojos se degrada fuera de distribución (cubo azul 5.1 cm, esfera verde
6.0 cm de proximidad de grasp vs 3.7 cm del cubo rojo).

Uso (CoppeliaSim en :23000):
    python experiments/collect_diffusion_dataset_v8_randomized.py [--n 1000]
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
from src.planning.diffusion_policy import DiffusionGraspPlanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("collect_v8rand")

DATASET_DIR = REPO / "data" / "datasets" / "sim_pick_v8rand"
SEED = 42
CHUNK_SIZE = 200
PARK = (-1.0, -1.0, -1.0)

FORMAS = ("cubo", "esfera", "cilindro")
PALETA = {
    "rojo": (0.85, 0.15, 0.15), "verde": (0.20, 0.75, 0.20),
    "azul": (0.15, 0.30, 0.85), "amarillo": (0.95, 0.78, 0.18),
    "naranja": (0.90, 0.50, 0.10), "blanco": (0.90, 0.90, 0.90),
}
COLORES = tuple(PALETA)


def _setup_piezas(sim) -> dict:
    """Cubo original de la escena + esfera y cilindro creados (estáticos)."""
    piezas = {"cubo": sim.getObject("/object_1")}
    for forma, tipo_attr, pure_code in (("esfera", "primitiveshape_spheroid", 1),
                                        ("cilindro", "primitiveshape_cylinder", 2)):
        try:
            h = sim.createPrimitiveShape(getattr(sim, tipo_attr), [0.05, 0.05, 0.05], 0)
        except Exception:
            h = sim.createPureShape(pure_code, 8, [0.05, 0.05, 0.05], 0.05)
        piezas[forma] = h
    return piezas


def _capture_randomizada(bridge, pose, piezas, forma: str, color: str) -> np.ndarray:
    sim = bridge.sim
    for k, (f, h) in enumerate(piezas.items()):
        sim.setObjectPosition(h, -1, [PARK[0] - 0.2 * k, PARK[1], PARK[2]])
    h = piezas[forma]
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, list(PALETA[color]))
    sim.setObjectPosition(h, -1, [float(p) for p in pose[:3, 3]])
    return _capture_rgbd_only(bridge)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    n_total = args.n

    from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out = DATASET_DIR / "heuristic.pt"

    start, poses, rgbds, trajs, formas_idx, colores_idx = 0, None, None, None, None, None
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
            start = done

    if poses is None:
        poses = np.zeros((n_total, 16), dtype=np.float32)
        rgbds = np.zeros((n_total, 4, 224, 224), dtype=np.float32)
        trajs = np.zeros((n_total, 16, 7), dtype=np.float32)
        formas_idx = np.zeros(n_total, dtype=np.int8)
        colores_idx = np.zeros(n_total, dtype=np.int8)

    planner = DiffusionGraspPlanner(action_dim=7, horizon=16, n_diffusion_steps=100, device="cpu")
    rng_pose = np.random.default_rng(SEED)
    rng_app = np.random.default_rng(SEED + 7)
    for _ in range(start):           # replay para resume determinista
        sample_pose(rng_pose)
        rng_app.integers(0, 3)
        rng_app.integers(0, len(COLORES))

    SCENE = REPO / "data" / "scenes" / "bin_base.ttt"
    i = start
    logger.info(f"Iter 8: {n_total - start} trayectorias con apariencia randomizada")

    while i < n_total:
        chunk_end = min(i + CHUNK_SIZE, n_total)
        with CoppeliaSimBridge() as bridge:
            bridge.load_scene(SCENE)
            piezas = _setup_piezas(bridge.sim)
            bridge.set_stepping(True)
            bridge.start_simulation()
            try:
                while i < chunk_end:
                    pose = sample_pose(rng_pose)
                    fi = int(rng_app.integers(0, 3))
                    ci = int(rng_app.integers(0, len(COLORES)))
                    rgbd = _capture_randomizada(bridge, pose, piezas, FORMAS[fi], COLORES[ci])
                    traj = planner.plan_grasp_heuristic(
                        pose, approach_distance=0.15, lift_height=0.10, with_deposit=True)
                    trajs[i], rgbds[i] = traj[0], rgbd
                    poses[i] = pose.flatten().astype(np.float32)
                    formas_idx[i], colores_idx[i] = fi, ci
                    i += 1
                    if i % 50 == 0:
                        logger.info(f"  {i}/{n_total}")
            finally:
                bridge.stop_simulation()

        torch.save({
            "poses": torch.from_numpy(poses), "rgbds": torch.from_numpy(rgbds),
            "trajs": torch.from_numpy(trajs),
            "formas": torch.from_numpy(formas_idx), "colores": torch.from_numpy(colores_idx),
            "formas_nombres": list(FORMAS), "colores_nombres": list(COLORES),
            "source": "heuristic_with_deposit_randomized", "seed": SEED, "n_valid": i,
        }, out)
        logger.info(f"  checkpoint: {i}/{n_total} ({out.stat().st_size / 1e6:.0f} MB)")

    # split 90/10 → train.pt / val.pt (mismo formato que v5)
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
    logger.info("Colección Iter 8 completa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
