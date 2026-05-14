#!/usr/bin/env python3
"""E2E LIVE con el modelo Diffusion Policy ULTRA (100ep, 10K trajs, hidden=256).

Variante de run_e2e_live.py que carga el modelo ultra-extendido.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Importar y monkey-patch el original con el modelo ultra
import experiments.run_e2e_live as original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-instances", type=int, default=30)
    parser.add_argument("--datasets", nargs="+", default=["ycbv", "tless"])
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--sim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"[e2e-live-ultra] n={args.n_instances}, ddim={args.ddim_steps}, sim={args.sim_steps}")

    sys.argv = [
        sys.argv[0],
        "--n-instances", str(args.n_instances),
        "--ddim-steps", str(args.ddim_steps),
        "--sim-steps", str(args.sim_steps),
        "--seed", str(args.seed),
    ]

    # Patch para cargar modelo ultra y dim 256
    import torch
    original_load = torch.load
    def patched_load(path, *a, **kw):
        ultra_path = REPO / "data/models/diffusion_policy_ultra.pth"
        if "diffusion_policy_grasp" in str(path) and ultra_path.exists():
            print(f"[patch] Cargando ULTRA: {ultra_path.name}")
            return original_load(ultra_path, *a, **kw)
        return original_load(path, *a, **kw)
    torch.load = patched_load

    # Tambien parchar el constructor ConditionalUNet1D para hidden_dim=256
    from src.planning import diffusion_policy as dp
    OriginalUNet = dp.ConditionalUNet1D
    class UltraUNet(OriginalUNet):
        def __init__(self, *a, **kw):
            kw["hidden_dim"] = 256
            super().__init__(*a, **kw)
    dp.ConditionalUNet1D = UltraUNet
    original.ConditionalUNet1D = UltraUNet  # tambien en el modulo importado

    original.main()

    # Renombrar output para no sobrescribir
    src = REPO / "experiments/results/pipeline_e2e/e2e_live_metrics.json"
    dst = REPO / "experiments/results/pipeline_e2e/e2e_live_ultra_metrics.json"
    if src.exists():
        import shutil
        shutil.copy(src, dst)
        print(f"\n[OK] Copia con modelo ULTRA: {dst.name}")


if __name__ == "__main__":
    main()
