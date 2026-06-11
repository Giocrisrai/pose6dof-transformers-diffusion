#!/usr/bin/env python3
"""Eval Iter 8 — robustez a apariencia: v7a (cubo rojo) vs v8 (randomizada).

Comparación PAREADA (mismas poses y misma secuencia de apariencias por seed):
  - política v7a_phase2 (entrenada solo con cubo rojo)
  - política v8_randomized (fine-tune con domain randomization)
sobre dos condiciones:
  - cubo rojo (in-distribution)
  - apariencia randomizada (forma de 3 × color de 6)

Uso (CoppeliaSim en :23000):
    python experiments/eval_diffusion_iter8_sim.py --n 25
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
from experiments.run_pick_with_diffusion import pick_with_dp
from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.planning.visual_encoder import ResNet18RGBDEncoder
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_v8")

OUT = REPO / "experiments/results/pick_with_diffusion"
SCENE = REPO / "data/scenes/bin_base.ttt"

FORMAS = ("cubo", "esfera", "cilindro")
PALETA = {
    "rojo": (0.85, 0.15, 0.15), "verde": (0.20, 0.75, 0.20),
    "azul": (0.15, 0.30, 0.85), "amarillo": (0.95, 0.78, 0.18),
    "naranja": (0.90, 0.50, 0.10), "blanco": (0.90, 0.90, 0.90),
}
COLORES = tuple(PALETA)


def _preparar_pieza(sim, forma: str, color: str) -> None:
    """Deja /object_1 con la forma y color pedidos (mismo truco que demo_charla)."""
    h = sim.getObject("/object_1")
    if forma != "cubo":
        sim.setObjectAlias(h, "object_1_off")
        sim.setObjectPosition(h, -1, [-1.0, -1.0, -1.0])
        try:
            tipo = (sim.primitiveshape_spheroid if forma == "esfera"
                    else sim.primitiveshape_cylinder)
            h = sim.createPrimitiveShape(tipo, [0.05, 0.05, 0.05], 0)
        except Exception:
            h = sim.createPureShape(1 if forma == "esfera" else 2, 8,
                                    [0.05, 0.05, 0.05], 0.05)
        sim.setObjectAlias(h, "object_1")
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 0)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
        try:
            sim.setShapeMass(h, 0.05)
        except Exception:
            pass
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, list(PALETA[color]))


def _load_policy(path: Path, device: str) -> DiffusionGraspPlanner:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    planner = DiffusionGraspPlanner(action_dim=7, horizon=16, n_diffusion_steps=100,
                                    device=device, hidden_dim=ckpt["config"]["hidden_dim"])
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    return planner


def _eval_condicion(planner, encoder, n: int, randomizada: bool) -> list[dict]:
    rng = np.random.default_rng(EVAL_SEED)
    rng_app = np.random.default_rng(77)          # misma secuencia para ambas políticas
    resultados = []
    for i in range(n):
        pose = sample_pose_eval(rng)
        if randomizada:
            forma = FORMAS[int(rng_app.integers(0, 3))]
            color = COLORES[int(rng_app.integers(0, len(COLORES)))]
        else:
            forma, color = "cubo", "rojo"
        try:
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                _preparar_pieza(bridge.sim, forma, color)
                r = pick_with_dp(planner, pose, bridge, frames_dir=None,
                                 visual_encoder=encoder, best_of_n=8)
            resultados.append({
                "i": i, "forma": forma, "color": color,
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


def _resumen(rs: list[dict]) -> dict:
    ok = [r for r in rs if "error" not in r]
    if not ok:
        return {"n": len(rs), "errores": len(rs)}
    e2e = [r["grasp_plausible"] and r["deposit_plausible"] and r["ik_converged"] for r in ok]
    return {
        "n": len(rs), "errores": len(rs) - len(ok),
        "grasp_pct": round(100 * np.mean([r["grasp_plausible"] for r in ok]), 1),
        "deposit_pct": round(100 * np.mean([r["deposit_plausible"] for r in ok]), 1),
        "ik_pct": round(100 * np.mean([r["ik_converged"] for r in ok]), 1),
        "pick_place_pct": round(100 * np.mean(e2e), 1),
        "mean_grasp_proximity_m": round(float(np.mean([r["grasp_proximity_m"] for r in ok])), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--torch-seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.torch_seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc_state = torch.load(REPO / "data/models/visual_encoder_iter5.pth",
                           map_location=device, weights_only=True)
    encoder = ResNet18RGBDEncoder(out_dim=enc_state["out_dim"]).to(device).eval()
    encoder.load_state_dict(enc_state["state_dict"])

    politicas = {
        "v7a_phase2": REPO / "data/models/diffusion_policy_v7a_phase2.pth",
        "v8_randomized": REPO / "data/models/diffusion_policy_v8_randomized.pth",
    }
    salida: dict = {"n": args.n, "torch_seed": args.torch_seed, "condiciones": {}}
    for nombre, path in politicas.items():
        if not path.exists():
            logger.warning(f"política {nombre} no existe ({path}); se omite")
            continue
        planner = _load_policy(path, device)
        for cond, rand in (("cubo_rojo", False), ("randomizada", True)):
            logger.info(f"═══ {nombre} × {cond} (n={args.n}) ═══")
            torch.manual_seed(args.torch_seed)
            rs = _eval_condicion(planner, encoder, args.n, rand)
            clave = f"{nombre}__{cond}"
            salida["condiciones"][clave] = {"resumen": _resumen(rs), "resultados": rs}
            logger.info(f"  → {salida['condiciones'][clave]['resumen']}")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "eval_v8_robustez.json"
    out_path.write_text(json.dumps(salida, indent=2))
    print("\n═══ RESUMEN ITER 8 ═══")
    for k, v in salida["condiciones"].items():
        print(f"  {k}: {v['resumen']}")
    print(f"  → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
