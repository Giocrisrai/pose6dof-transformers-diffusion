#!/usr/bin/env python3
"""Eval Iter 9 — selección del objetivo con distractores en escena (clutter).

La escena ya no tiene una sola pieza: se añaden distractores de forma/color
aleatorios cerca del objetivo. La política recibe la pose del OBJETIVO (como
siempre — en el pipeline completo viene de la estimación 6-DoF) y debe ir por
esa pieza ignorando las demás. Lo que cambia respecto a las evals previas es el
conditioning visual: el encoder ve una escena con varias piezas.

Comparación PAREADA (mismas poses, apariencias y distractores por seed):
  - condición sin_distractores (igual que Iter 8 randomizada)
  - condición con_distractores (2 piezas extra de apariencia aleatoria)

Uso (CoppeliaSim en :23000):
    python experiments/eval_diffusion_iter9_sim.py --n 25 [--politicas v8_randomized]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.eval_diffusion_iter2_sim import EVAL_SEED, sample_pose_eval
from experiments.eval_diffusion_iter8_sim import (
    COLORES, FORMAS, PALETA, _load_policy, _preparar_pieza, _resumen,
)
from experiments.run_pick_with_diffusion import pick_with_dp
from src.planning.visual_encoder import ResNet18RGBDEncoder
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_v9")

OUT = REPO / "experiments/results/pick_with_diffusion"
SCENE = REPO / "data/scenes/bin_base.ttt"

N_DISTRACTORES = 2
SEP_MIN_M = 0.09          # entre centros (piezas de 5 cm)
# región de distractores: el workspace de eval ampliado un poco (siguen en mesa
# y en el campo de la cámara)
DIST_X = (0.38, 0.58)
DIST_Y = (-0.19, -0.01)
Z_PIEZA = 0.033


def _posiciones_distractores(rng: np.random.Generator, target_xy) -> list[tuple]:
    """Rejection sampling: N posiciones separadas ≥ SEP_MIN del target y entre sí."""
    puestos = [tuple(target_xy)]
    out = []
    while len(out) < N_DISTRACTORES:
        x = float(rng.uniform(*DIST_X))
        y = float(rng.uniform(*DIST_Y))
        if all((x - px) ** 2 + (y - py) ** 2 >= SEP_MIN_M**2 for px, py in puestos):
            puestos.append((x, y))
            out.append((x, y))
    return out


def _crear_distractor(sim, forma: str, color: str, xy: tuple) -> None:
    try:
        if forma == "cubo":
            h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [0.05, 0.05, 0.05], 0)
        else:
            tipo = (sim.primitiveshape_spheroid if forma == "esfera"
                    else sim.primitiveshape_cylinder)
            h = sim.createPrimitiveShape(tipo, [0.05, 0.05, 0.05], 0)
    except Exception:
        h = sim.createPureShape({"cubo": 0, "esfera": 1, "cilindro": 2}[forma], 8,
                                [0.05, 0.05, 0.05], 0.05)
    sim.setObjectAlias(h, "distractor")
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
    try:
        sim.setShapeMass(h, 0.05)
    except Exception:
        pass
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, list(PALETA[color]))
    sim.setObjectPosition(h, -1, [xy[0], xy[1], Z_PIEZA])


def _eval_condicion(planner, encoder, n: int, con_distractores: bool) -> list[dict]:
    rng = np.random.default_rng(EVAL_SEED)
    rng_app = np.random.default_rng(77)           # misma secuencia que Iter 8
    rng_dist = np.random.default_rng(EVAL_SEED + 123)
    resultados = []
    for i in range(n):
        pose = sample_pose_eval(rng)
        forma = FORMAS[int(rng_app.integers(0, 3))]
        color = COLORES[int(rng_app.integers(0, len(COLORES)))]
        # consumir SIEMPRE el stream de distractores → pareado entre condiciones
        pos_dist = _posiciones_distractores(rng_dist, pose[:2, 3])
        app_dist = [(FORMAS[int(rng_dist.integers(0, 3))],
                     COLORES[int(rng_dist.integers(0, len(COLORES)))])
                    for _ in range(N_DISTRACTORES)]
        try:
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                _preparar_pieza(bridge.sim, forma, color)
                if con_distractores:
                    for (fd, cd), xy in zip(app_dist, pos_dist):
                        _crear_distractor(bridge.sim, fd, cd, xy)
                r = pick_with_dp(planner, pose, bridge, frames_dir=None,
                                 visual_encoder=encoder, best_of_n=8)
            resultados.append({
                "i": i, "forma": forma, "color": color,
                "distractores": [{"forma": f, "color": c, "x": round(x, 3), "y": round(y, 3)}
                                 for (f, c), (x, y) in zip(app_dist, pos_dist)]
                if con_distractores else [],
                "grasp_proximity_m": r["grasp_proximity_m"],
                "deposit_error_m": r["deposit_error_m"],
                "ik_converged": r["ik_converged"],
                "grasp_plausible": r["grasp_plausible"],
                "deposit_plausible": r["deposit_plausible"],
            })
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  pick {i} falló: {e}")
            resultados.append({"i": i, "forma": forma, "color": color, "error": str(e)})
        if (i + 1) % 5 == 0:
            logger.info(f"  {i + 1}/{n}")
    return resultados


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--torch-seed", type=int, default=2026)
    parser.add_argument("--politicas", nargs="+", default=["v8_randomized"])
    parser.add_argument("--condiciones", nargs="+",
                        default=["sin_distractores", "con_distractores"])
    args = parser.parse_args()

    torch.manual_seed(args.torch_seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    def _cargar_encoder(nombre_enc: str) -> ResNet18RGBDEncoder:
        enc_state = torch.load(REPO / f"data/models/{nombre_enc}.pth",
                               map_location=device, weights_only=True)
        enc = ResNet18RGBDEncoder(out_dim=enc_state["out_dim"]).to(device).eval()
        enc.load_state_dict(enc_state["state_dict"])
        return enc

    # cada política con el encoder de SU entrenamiento (ver Iter 8)
    politicas = {
        "v8_randomized": (REPO / "data/models/diffusion_policy_v8_randomized.pth",
                          "visual_encoder_iter8rand"),
        "v9_clutter": (REPO / "data/models/diffusion_policy_v9_clutter.pth",
                       "visual_encoder_iter9clut"),
    }
    out_path = OUT / "eval_v9_clutter.json"
    salida: dict = {"n": args.n, "torch_seed": args.torch_seed,
                    "n_distractores": N_DISTRACTORES, "condiciones": {}}
    if out_path.exists():
        salida = json.loads(out_path.read_text())
        salida["condiciones"] = salida.get("condiciones", {})
    for nombre, (path, enc_nombre) in politicas.items():
        if nombre not in args.politicas:
            continue
        if not path.exists():
            logger.warning(f"política {nombre} no existe ({path}); se omite")
            continue
        encoder = _cargar_encoder(enc_nombre)
        planner = _load_policy(path, device)
        for cond, con_d in (("sin_distractores", False), ("con_distractores", True)):
            if cond not in args.condiciones:
                continue
            logger.info(f"═══ {nombre} × {cond} (n={args.n}) ═══")
            torch.manual_seed(args.torch_seed)
            rs = _eval_condicion(planner, encoder, args.n, con_d)
            clave = f"{nombre}__{cond}"
            salida["condiciones"][clave] = {"resumen": _resumen(rs), "resultados": rs}
            logger.info(f"  → {salida['condiciones'][clave]['resumen']}")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(salida, indent=2))
    print("\n═══ RESUMEN ITER 9 (clutter) ═══")
    for k, v in salida["condiciones"].items():
        print(f"  {k}: {v['resumen']}")
    print(f"  → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
