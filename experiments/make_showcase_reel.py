# experiments/make_showcase_reel.py
"""Genera reel_showcase.mp4: hero pick cinematográfico (Iter 7c) + valor."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.eval_diffusion_iter2_sim import EVAL_SEED, sample_pose_eval
from experiments.run_pick_with_diffusion import compile_mp4, pick_with_dp
from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.planning.visual_encoder import ResNet18RGBDEncoder
from src.simulation.cine_camera import CineCamera
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.reel_overlay import make_title_card

SCENE = REPO / "data/scenes/bin_base.ttt"
OUT = REPO / "experiments/results/demo_reel"
WORKSPACE_CENTER = (0.0, -0.30, 0.10)  # centro aproximado bin/deposit
POSE_INDEX = 49
TORCH_SEED = 3
# 16 waypoints × (n_substeps=8 × steps_per_substep=2 + settle=30 por waypoint) = 736
# (verificado: el pick captura 736 frames; progress = i/TOTAL recorre todo el clip)
TOTAL_FRAMES_ESTIM = 16 * (8 * 2 + 30)


def run_hero_pick(frames_dir: Path) -> dict:
    torch.manual_seed(TORCH_SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(REPO / "data/models/diffusion_policy_v7a_phase2.pth",
                      map_location=device, weights_only=True)
    planner = DiffusionGraspPlanner(action_dim=7, horizon=16, n_diffusion_steps=100,
                                    device=device, hidden_dim=ckpt["config"]["hidden_dim"])
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    es = torch.load(REPO / "data/models/visual_encoder_iter5.pth",
                    map_location=device, weights_only=True)
    enc = ResNet18RGBDEncoder(out_dim=es["out_dim"]).to(device).eval()
    enc.load_state_dict(es["state_dict"])

    rng = np.random.default_rng(EVAL_SEED)
    pose = None
    for _ in range(POSE_INDEX + 1):
        pose = sample_pose_eval(rng)

    with CoppeliaSimBridge() as bridge:
        bridge.load_scene(SCENE)
        cam = CineCamera(bridge)
        cam.create()
        try:
            tip = bridge.sim.getObject("/tip")
            state = {"i": 0}

            def hook():
                tcp = tuple(bridge.sim.getObjectPosition(tip, -1))
                progress = min(1.0, state["i"] / TOTAL_FRAMES_ESTIM)
                cam.aim(progress, tcp, WORKSPACE_CENTER)
                cam.capture(frames_dir, state["i"])
                state["i"] += 1

            result = pick_with_dp(planner, pose, bridge, frames_dir=None,
                                  visual_encoder=enc, best_of_n=8, frame_hook=hook)
        finally:
            cam.remove()
    result["frames_captured"] = state["i"]
    # Guarda contra drift: si los constantes internas de pick_with_dp cambian,
    # TOTAL_FRAMES_ESTIM dejaría de coincidir y la cámara se congelaría/overshooting.
    drift = abs(result["frames_captured"] - TOTAL_FRAMES_ESTIM)
    assert drift <= 46, (
        f"frames_captured={result['frames_captured']} difiere de "
        f"TOTAL_FRAMES_ESTIM={TOTAL_FRAMES_ESTIM} en {drift} (>46); "
        f"actualizá TOTAL_FRAMES_ESTIM si cambiaron los internals de pick_with_dp"
    )
    return result


# Cada tupla es (texto, escala_de_fuente) — la escala la consume make_title_card
# (NO es duración). La duración por tarjeta es fija (SECONDS_PER_CARD).
VALUE_CARDS = [
    [("Sistema bin-picking 6-DoF", 1.3),
     ("FoundationPose  +  Diffusion Policy", 0.7)],
    [("Lo que se logro", 1.1),
     ("pick-and-place E2E 84%", 0.8),
     ("IK 100%  -  ciclo p95 < 10 s", 0.7)],
    [("Por que es mejor", 1.1),
     ("corre en hardware accesible (~USD 1.920)", 0.7),
     ("vs setups industriales USD 15k-150k", 0.7)],
    [("Aplicaciones", 1.1),
     ("logistica / e-commerce  -  manufactura  -  clasificacion", 0.65)],
    [("Honestidad declarada", 1.0),
     ("grasp por snap+attach (estandar en sims)", 0.7),
     ("siguiente: grasp fisico + robot real", 0.65)],
]
SECONDS_PER_CARD = 4.0


def build_value_clip(frames_dir: Path, fps: int = 30) -> int:
    """Genera frames de las tarjetas de valor (SECONDS_PER_CARD c/u).
    Devuelve nº de frames escritos."""
    from PIL import Image
    frames_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    n_per_card = int(round(SECONDS_PER_CARD * fps))
    for lines in VALUE_CARDS:
        card = make_title_card(lines)  # (H,W,3) uint8; el float de cada línea = escala
        for _ in range(n_per_card):
            Image.fromarray(card).save(frames_dir / f"{idx:06d}.png")
            idx += 1
    return idx


def main() -> int:
    fps = 30
    frames_a = OUT / "frames_showcase"
    import shutil
    shutil.rmtree(frames_a, ignore_errors=True)
    result = run_hero_pick(frames_a)
    assert result["grasp_plausible"] and result["deposit_plausible"] and result["ik_converged"], (
        f"pick no limpio: grasp_plausible={result['grasp_plausible']} "
        f"deposit_plausible={result['deposit_plausible']} ik={result['ik_converged']}"
    )

    part_a = OUT / "_showcase_partA.mp4"
    compile_mp4(frames_a, part_a, fps=fps)
    frames_b = OUT / "frames_value"
    shutil.rmtree(frames_b, ignore_errors=True)
    build_value_clip(frames_b, fps=fps)
    part_b = OUT / "_showcase_partB.mp4"
    compile_mp4(frames_b, part_b, fps=fps)

    import subprocess
    listf = OUT / "_showcase_concat.txt"
    listf.write_text(f"file '{part_a.resolve()}'\nfile '{part_b.resolve()}'\n")
    final = OUT / "reel_showcase.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listf),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(final)],
                   check=True, capture_output=True)
    for p in (part_a, part_b, listf):
        p.unlink(missing_ok=True)
    print(f"reel_showcase.mp4 listo: {final}")
    print(f"  hero frames={result['frames_captured']} grasp={result['grasp_proximity_m']*100:.1f}cm "
          f"deposit={result['deposit_error_m']*100:.1f}cm ik={result['ik_converged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
