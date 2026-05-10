#!/usr/bin/env python3
"""Construye una escena custom de CoppeliaSim con piezas BOP reales.

Importa los meshes CAD de T-LESS o YCB-Video como objetos rigidos en una
escena con bin (contenedor) + camara cenital + brazo robotico Ragnar.

Uso:
    1. Lanzar CoppeliaSim Edu V4.10 (open -a CoppeliaSim_Edu)
    2. python scripts/build_custom_bop_scene.py --dataset tless --n-objects 5

Salidas:
    experiments/results/custom_scenes/scene_<dataset>.ttt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCENES_DIR = REPO / "experiments/results/custom_scenes"
SCENES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tless", "ycbv"], default="tless")
    parser.add_argument("--n-objects", type=int, default=5)
    parser.add_argument("--bin-x", type=float, default=0.4)
    parser.add_argument("--bin-y", type=float, default=0.4)
    parser.add_argument("--bin-z", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[build-scene] dataset={args.dataset}, n_objects={args.n_objects}")

    # Conectar a CoppeliaSim
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError:
        print("[FAIL] instalar coppeliasim_zmqremoteapi_client primero")
        return 1

    client = RemoteAPIClient(host="localhost", port=23000)
    sim = client.getObject("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        time.sleep(0.3)

    # Crear escena nueva
    sim.closeScene()
    print("  Escena vacia creada")

    # Crear plano (suelo)
    floor = sim.createPrimitiveShape(
        sim.primitiveshape_plane,
        [2.0, 2.0, 0.005],
    )
    sim.setObjectAlias(floor, "Floor")
    sim.setObjectPosition(floor, -1, [0.0, 0.0, 0.0])
    print(f"  Floor creado: handle={floor}")

    # Crear bin (contenedor) - 4 paredes + base
    bin_handles = []
    base = sim.createPrimitiveShape(
        sim.primitiveshape_cuboid,
        [args.bin_x, args.bin_y, 0.005],
    )
    sim.setObjectAlias(base, "Bin_base")
    sim.setObjectPosition(base, -1, [0.5, 0.0, args.bin_z / 2])
    bin_handles.append(base)

    wall_thickness = 0.01
    walls = [
        ("Bin_wall_X+", [wall_thickness, args.bin_y, args.bin_z], [0.5 + args.bin_x/2, 0, args.bin_z]),
        ("Bin_wall_X-", [wall_thickness, args.bin_y, args.bin_z], [0.5 - args.bin_x/2, 0, args.bin_z]),
        ("Bin_wall_Y+", [args.bin_x, wall_thickness, args.bin_z], [0.5, args.bin_y/2, args.bin_z]),
        ("Bin_wall_Y-", [args.bin_x, wall_thickness, args.bin_z], [0.5, -args.bin_y/2, args.bin_z]),
    ]
    for name, size, pos in walls:
        h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size)
        sim.setObjectAlias(h, name)
        sim.setObjectPosition(h, -1, pos)
        bin_handles.append(h)
    print(f"  Bin creado con {len(bin_handles)} elementos")

    # Importar meshes BOP
    DATA_DIR = REPO / f"data/datasets/{args.dataset}"
    if args.dataset == "tless":
        models_dir = DATA_DIR / "models_cad"
    else:
        models_dir = DATA_DIR / "models"

    if not models_dir.exists():
        print(f"  [WARN] models dir no existe: {models_dir}")
        print("         Continuando sin objetos importados (escena vacia se guardara)")
    else:
        ply_files = sorted(models_dir.glob("obj_*.ply"))[:args.n_objects]
        print(f"  Encontrados {len(ply_files)} archivos .ply")

        rng = np.random.default_rng(args.seed)
        for i, ply in enumerate(ply_files):
            try:
                # Importar como mesh
                obj_handle = sim.importShape(0, str(ply), 0, 0.0001, 1.0)
                sim.setObjectAlias(obj_handle, f"BOP_obj_{i:02d}_{ply.stem}")
                # Posicionar en el bin con dispersion aleatoria
                x = 0.5 + rng.uniform(-args.bin_x/3, args.bin_x/3)
                y = rng.uniform(-args.bin_y/3, args.bin_y/3)
                z = args.bin_z + 0.05 + i * 0.04  # apilados gradualmente
                sim.setObjectPosition(obj_handle, -1, [x, y, z])
                # Escala (T-LESS y YCB estan en mm, queremos m)
                sim.setObjectInt32Param(obj_handle, sim.shapeintparam_static, 0)  # dinamico
                sim.setObjectInt32Param(obj_handle, sim.shapeintparam_respondable, 1)
                print(f"    [{i+1}] {ply.name}: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            except Exception as e:
                print(f"    [warn] {ply.name}: {e}")

    # Vision sensor cenital
    vs = sim.createVisionSensor(
        1,
        [800, 600, 0, 0],
        [0.01, 5.0, 1.0472, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.0, 0.0],
    )
    sim.setObjectAlias(vs, "tfm_bin_overview_sensor")
    sim.setObjectPosition(vs, -1, [0.5, 0.0, 1.5])
    sim.setObjectOrientation(vs, -1, [3.14159, 0.0, 0.0])
    print(f"  Vision sensor creado: handle={vs}")

    # Guardar escena
    out_path = SCENES_DIR / f"scene_{args.dataset}_n{args.n_objects}.ttt"
    sim.saveScene(str(out_path))
    print(f"\n[OK] Escena guardada: {out_path}")
    print(f"     Para usar: sim.loadScene('{out_path}')")


if __name__ == "__main__":
    sys.exit(main() or 0)
