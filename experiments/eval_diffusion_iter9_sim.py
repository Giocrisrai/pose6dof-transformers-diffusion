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

from experiments.collect_diffusion_dataset_v9_clutter import park_piezas_fijas
from experiments.eval_diffusion_iter2_sim import EVAL_SEED, sample_pose_eval
from experiments.eval_diffusion_iter8_sim import (
    COLORES,
    FORMAS,
    PALETA,
    _load_policy,
    _preparar_pieza,
    _resumen,
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

MAX_INTENTOS = 400


def _posiciones_distractores(rng: np.random.Generator, target_xy) -> list[tuple]:
    """Rejection sampling: N posiciones separadas ≥ SEP_MIN del target y entre
    sí (con tope de intentos por seguridad). Las piezas fijas de la escena se
    estacionan fuera (park_piezas_fijas), así que no entran en la colocación."""
    puestos = [tuple(target_xy)]
    out = []
    for _ in range(MAX_INTENTOS):
        if len(out) >= N_DISTRACTORES:
            break
        x = float(rng.uniform(*DIST_X))
        y = float(rng.uniform(*DIST_Y))
        if all((x - px) ** 2 + (y - py) ** 2 >= SEP_MIN_M**2 for px, py in puestos):
            puestos.append((x, y))
            out.append((x, y))
    return out


def _crear_distractor(sim, forma: str, color: str, xy: tuple) -> int:
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
    return h


def _eval_condicion(planner, encoder, n: int, con_distractores: bool,
                    seguro: bool = False) -> list[dict]:
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
                park_piezas_fijas(bridge.sim)
                _preparar_pieza(bridge.sim, forma, color)
                handles = []
                if con_distractores:
                    handles = [_crear_distractor(bridge.sim, fd, cd, xy)
                               for (fd, cd), xy in zip(app_dist, pos_dist)]
                obstaculos = ([[x, y, Z_PIEZA] for x, y in pos_dist]
                              if (con_distractores and seguro) else None)
                r = pick_with_dp(planner, pose, bridge, frames_dir=None,
                                 visual_encoder=encoder, best_of_n=8,
                                 obstacles=obstaculos, track_handles=handles)
            # ¿el brazo empujó alguna pieza que no era el objetivo?
            despl = [float(np.hypot(p[0] - x, p[1] - y))
                     for p, (x, y) in zip(r["tracked_end"], pos_dist)]
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
                "desplazamiento_distractores_m": [round(d, 4) for d in despl],
                "n_candidates_unsafe": r["n_candidates_unsafe"],
                "clearance_m_selected": r["clearance_m_selected"],
            })
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  pick {i} falló: {e}")
            resultados.append({"i": i, "forma": forma, "color": color, "error": str(e)})
        if (i + 1) % 5 == 0:
            logger.info(f"  {i + 1}/{n}")
    return resultados


UMBRAL_EMPUJON_M = 0.02


def _resumen_v9(rs: list[dict]) -> dict:
    """Resumen base de Iter 8 + métricas de clutter (empujones y seguridad).

    Los empujones cuentan cuánto se movió el distractor más desplazado en
    cada pick (las piezas fijas se estacionan fuera, no participan)."""
    res = _resumen(rs)
    ok = [r for r in rs if "error" not in r and r.get("distractores")]
    if ok:
        maxd = [max(r["desplazamiento_distractores_m"]) for r in ok
                if r["desplazamiento_distractores_m"]]
        if maxd:
            # binaria (robusta): ¿el brazo movió alguna pieza no-objetivo >2cm?
            res["empujadas_pct"] = round(
                100 * float(np.mean([d > UMBRAL_EMPUJON_M for d in maxd])), 1)
            # MEDIANA, no media: esferas/cilindros ruedan fuera del bin y dan
            # outliers de metros (sin fricción) que harían inútil la media
            res["mediana_max_desplazamiento_cm"] = round(100 * float(np.median(maxd)), 2)
        descartes = [r["n_candidates_unsafe"] for r in ok]
        if any(descartes):
            res["mean_candidatos_descartados"] = round(float(np.mean(descartes)), 2)
    return res


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--torch-seed", type=int, default=2026)
    parser.add_argument("--politicas", nargs="+", default=["v8_randomized"])
    parser.add_argument("--condiciones", nargs="+",
                        default=["sin_distractores", "con_distractores",
                                 "con_distractores_seguro"])
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
        for cond, con_d, seguro in (("sin_distractores", False, False),
                                    ("con_distractores", True, False),
                                    ("con_distractores_seguro", True, True)):
            if cond not in args.condiciones:
                continue
            logger.info(f"═══ {nombre} × {cond} (n={args.n}) ═══")
            torch.manual_seed(args.torch_seed)
            rs = _eval_condicion(planner, encoder, args.n, con_d, seguro)
            clave = f"{nombre}__{cond}"
            salida["condiciones"][clave] = {"resumen": _resumen_v9(rs), "resultados": rs}
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
