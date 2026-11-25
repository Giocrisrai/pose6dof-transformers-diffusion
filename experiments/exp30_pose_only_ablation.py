#!/usr/bin/env python3
"""Exp 30 — ¿aporta la imagen a la política de agarre? Ablación solo pose.

La política v9 recibe la pose de la pieza (12 valores) y un embedding de la
imagen RGB-D (52 valores). La exploración 15 mostró que ese embedding la vuelve
frágil: si cambia la postura del brazo o aparece un objeto en la imagen, apunta
peor. Aquí se entrena la misma red con los mismos 900 pares (pose, trayectoria)
de v9, pero sin imagen, y se mide el error de agarre sobre los 100 de validación.

Los datos (`data/pose_only_v9/`) son las poses y trayectorias del dataset de v9;
no incluyen imágenes, por eso caben en el repositorio.

Uso:
    python experiments/exp30_pose_only_ablation.py            # evalúa el checkpoint
    python experiments/exp30_pose_only_ablation.py --train    # reentrena (unos 5 min en MPS)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner

DATOS = REPO / "data/pose_only_v9"
PESOS = REPO / "data/models/diffusion_policy_v9_pose_only.pth"
OUT = REPO / "experiments/results/exp30_pose_only"
DEPOSITO_XY = (-0.30, -0.30)
UMBRAL_AGARRE_M = 0.05
MEJOR_DE_N = 8


def entrenar(epochs: int) -> None:
    subprocess.run(
        [sys.executable, str(REPO / "experiments/train_diffusion_on_sim.py"), "--from-scratch",
         "--hidden-dim", "256", "--epochs", str(epochs), "--dataset-dir", str(DATOS),
         "--checkpoint-out", str(PESOS)],
        check=True,
    )
    # el checkpoint versionado no necesita el estado del optimizador
    ckpt = torch.load(PESOS, map_location="cpu", weights_only=True)
    ckpt.pop("optimizer_state_dict", None)
    torch.save(ckpt, PESOS)


def evaluar(seed: int) -> dict:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(PESOS, map_location=device, weights_only=True)
    planner = DiffusionGraspPlanner(action_dim=7, horizon=16, n_diffusion_steps=100,
                                    device=device, hidden_dim=ckpt["config"]["hidden_dim"])
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()

    val = torch.load(DATOS / "val.pt", weights_only=True)
    torch.manual_seed(seed)
    error_agarre, error_deposito = [], []
    for cond in val["conds"]:
        pose = np.eye(4)
        pose[:3, :] = cond[:12].numpy().reshape(3, 4)
        candidatas = np.asarray(planner.plan_grasp(pose, n_samples=MEJOR_DE_N))
        objetivo = pose[:3, 3]
        mejor, mejor_error = None, np.inf
        for tr in candidatas:
            k = next((i for i in range(len(tr)) if tr[i, 6] < 0.5), 8)
            e = float(np.linalg.norm(objetivo - tr[k, :3]))
            if e < mejor_error:
                mejor, mejor_error = tr, e
        assert mejor is not None
        error_agarre.append(mejor_error)
        k = next((i for i in range(len(mejor)) if mejor[i, 6] < 0.5), 8)
        suelta = next((i for i in range(k, len(mejor)) if mejor[i, 6] > 0.5), len(mejor) - 1)
        error_deposito.append(float(np.hypot(mejor[suelta, 0] - DEPOSITO_XY[0],
                                             mejor[suelta, 1] - DEPOSITO_XY[1])))
    e = np.array(error_agarre)
    return {
        "n": len(e),
        "seed": seed,
        "mejor_de_n": MEJOR_DE_N,
        "error_agarre_medio_cm": round(100 * float(e.mean()), 2),
        "agarres_bajo_5cm_pct": round(100 * float((e < UMBRAL_AGARRE_M).mean()), 1),
        "error_deposito_mediano_cm": round(100 * float(np.median(error_deposito)), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="reentrena antes de evaluar")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.train:
        entrenar(args.epochs)

    OUT.mkdir(parents=True, exist_ok=True)
    salida_path = OUT / "exp30_results.json"
    salida = json.loads(salida_path.read_text()) if salida_path.exists() else {}
    salida["solo_pose_validacion"] = evaluar(args.seed)
    salida_path.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(json.dumps(salida["solo_pose_validacion"], indent=2, ensure_ascii=False))
    print(f"→ {salida_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
