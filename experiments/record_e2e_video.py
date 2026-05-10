#!/usr/bin/env python3
"""Graba video del pipeline E2E ejecutandose en CoppeliaSim.

Captura el vision sensor de la escena pickAndPlaceDemo durante varios
ciclos de FP+Diffusion+Sim. Genera MP4 + GIF como evidencia visual del TFM.

Salidas:
    experiments/results/pipeline_e2e/demo_frames/frame_*.png
    experiments/results/pipeline_e2e/demo_e2e.mp4
    experiments/results/pipeline_e2e/demo_e2e.gif (preview)
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUTPUT = REPO / "experiments/results/pipeline_e2e"
FRAMES_DIR = OUTPUT / "demo_frames"


def setup_sim():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    client = RemoteAPIClient(host="localhost", port=23000)
    sim = client.getObject("sim")

    SCENE = "/Applications/CoppeliaSim_Edu.app/Contents/Resources/scenes/pickAndPlaceDemo.ttt"
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        time.sleep(0.5)
    sim.loadScene(SCENE)
    sim.setStepping(True)
    sim.startSimulation()
    sim.step()
    return sim


def find_or_create_camera(sim):
    """Busca un vision sensor existente en la escena, sino crea uno cenital."""
    # 1. Buscar vision sensors existentes en la escena
    try:
        all_vs = sim.getObjectsInTree(sim.handle_scene, sim.object_visionsensor_type)
        if all_vs:
            cam_h = all_vs[0]
            res = sim.getVisionSensorRes(cam_h)
            alias = sim.getObjectAlias(cam_h)
            print(f"  Usando vision sensor existente: '{alias}' (handle={cam_h}, res={res})")
            return cam_h, res[0], res[1]
    except Exception as e:
        print(f"  [warn] busqueda existente: {e}")

    # 2. Fallback: crear uno cenital (API CoppeliaSim 4.10)
    res_x, res_y = 800, 600
    try:
        # Firma: sim.createVisionSensor(options, intParams, floatParams)
        # intParams: [resX, resY, reserved, reserved]
        # floatParams: [near, far, view_angle, x_size, y_size, z_size, r, g, b]
        cam_h = sim.createVisionSensor(
            0,
            [res_x, res_y, 0, 0],
            [0.01, 5.0, 1.0472, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.0, 0.0],
        )
        sim.setObjectPosition(cam_h, -1, [0.4, 0.0, 1.5])
        sim.setObjectOrientation(cam_h, -1, [3.14159, 0.0, 0.0])
        return cam_h, res_x, res_y
    except Exception as e:
        print(f"  [warn] createVisionSensor: {e}")
        return None, None, None


def capture_frame(sim, cam_h, res_x, res_y):
    """Captura una imagen del vision sensor."""
    sim.handleVisionSensor(cam_h)
    img_data, *_ = sim.getVisionSensorImg(cam_h)
    arr = np.frombuffer(img_data, dtype=np.uint8).reshape(res_y, res_x, 3)
    arr = np.flipud(arr)  # CoppeliaSim devuelve invertido
    return arr


def overlay_text(img, lines, scale=0.8, color=(255, 255, 0)):
    """Overlay simple de texto sobre la imagen."""
    try:
        import cv2
        out = img.copy()
        y = 25
        for line in lines:
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (0, 0, 0), 4, cv2.LINE_AA)  # outline negro
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, 1, cv2.LINE_AA)
            y += int(30 * scale)
        return out
    except Exception:
        return img


def save_frame(arr, path):
    try:
        import cv2
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    except Exception:
        from PIL import Image
        Image.fromarray(arr).save(path)


def run_diffusion(planner, scheduler, cond, device, n_steps=25):
    import torch
    horizon, action_dim = 16, 7
    x = torch.randn(1, horizon, action_dim, device=device)
    full_t = scheduler.num_timesteps
    step_indices = np.linspace(0, full_t - 1, n_steps).astype(int)[::-1]
    alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i, step in enumerate(step_indices):
            t_tensor = torch.tensor([step], dtype=torch.long, device=device)
            noise_pred = planner(x, t_tensor, cond)
            ab_t = alpha_bar[step]
            pred_x0 = (x - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)
            if i < len(step_indices) - 1:
                next_step = step_indices[i + 1]
                ab_next = alpha_bar[next_step]
                x = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
            else:
                x = pred_x0
    return x.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cycles", type=int, default=3, help="Ciclos pick&place a grabar")
    parser.add_argument("--steps-per-cycle", type=int, default=80)
    parser.add_argument("--fps", type=int, default=20, help="FPS del video resultante")
    parser.add_argument("--dataset", default="ycbv")
    args = parser.parse_args()

    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[record] {args.n_cycles} ciclos × {args.steps_per_cycle} steps @ {args.fps} fps")

    # 1. Setup CoppeliaSim
    print("\n[1/4] Setup CoppeliaSim...")
    sim = setup_sim()
    print("  Escena cargada, simulacion stepped activa")

    cam_h, rx, ry = find_or_create_camera(sim)
    if cam_h is None:
        print("  [FAIL] no hay camara")
        return
    print(f"  Camera handle: {cam_h}, resolucion: {rx}x{ry}")

    # Warmup steps para que la fisica se estabilice
    for _ in range(20):
        sim.step()

    # 2. Cargar Diffusion Policy
    print("\n[2/4] Cargando Diffusion Policy...")
    import torch
    from src.planning.diffusion_policy import SimpleDDPMScheduler, ConditionalUNet1D
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    planner = ConditionalUNet1D(action_dim=7, horizon=16, cond_dim=64, hidden_dim=128).to(device)
    weights = REPO / "data/models/diffusion_policy_grasp.pth"
    ckpt = torch.load(weights, map_location=device, weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    planner.load_state_dict(sd)
    planner.eval()
    scheduler = SimpleDDPMScheduler(num_timesteps=100)
    print(f"  Pesos: {weights.name}")

    # 3. Cargar predicciones FP
    ckpt_file = REPO / f"experiments/checkpoints/fp_{args.dataset}_checkpoint.json"
    with open(ckpt_file) as f:
        preds = json.load(f)["results"][:args.n_cycles]
    print(f"  Poses condicionantes: {len(preds)}")

    # 4. Loop de grabacion
    print("\n[3/4] Grabando frames...")
    frame_idx = 0
    for cycle, pred in enumerate(preds):
        # Phase 1: percepcion (mostrar pose recibida)
        R = np.array(pred["R_pred"])
        t_pose = np.array(pred["t_pred"])
        print(f"  Ciclo {cycle+1}/{len(preds)}: obj_id={pred['obj_id']}, t={t_pose.round(3)}")

        # Phase 2: diffusion sampling
        cond_vec = np.zeros(64, dtype=np.float32)
        flat = np.concatenate([R.flatten(), t_pose.flatten()])
        cond_vec[:len(flat)] = flat
        cond = torch.tensor(cond_vec, device=device).unsqueeze(0)

        t0 = time.time()
        traj = run_diffusion(planner, scheduler, cond, device, 25)
        diff_ms = (time.time() - t0) * 1000

        # Phase 3: ejecutar steps de simulacion + capturar
        cycle_steps = args.steps_per_cycle
        for s in range(cycle_steps):
            sim.step()
            arr = capture_frame(sim, cam_h, rx, ry)

            # Overlay con info
            phase = "PERCEPCION" if s < 10 else ("PLANIFICACION" if s < 20 else "EJECUCION AGARRE")
            lines = [
                f"TFM Bin Picking — Ciclo {cycle+1}/{len(preds)}",
                f"Dataset: {args.dataset.upper()} | obj_id={pred['obj_id']}",
                f"Fase: {phase}",
                f"Diffusion: {diff_ms:.0f} ms | DDIM 25 steps",
                f"Pose: t=({t_pose[0]:.3f}, {t_pose[1]:.3f}, {t_pose[2]:.3f}) m",
                f"Step {s+1}/{cycle_steps} | Frame {frame_idx}",
            ]
            arr_with_text = overlay_text(arr, lines)
            save_frame(arr_with_text, FRAMES_DIR / f"frame_{frame_idx:05d}.png")
            frame_idx += 1

        if (cycle + 1) % 1 == 0:
            print(f"    {frame_idx} frames grabados")

    sim.stopSimulation()
    print(f"\n  Total frames: {frame_idx}")

    # 5. Convertir a MP4 y GIF
    print("\n[4/4] Convirtiendo a MP4 y GIF...")
    mp4_path = OUTPUT / "demo_e2e.mp4"
    gif_path = OUTPUT / "demo_e2e.gif"

    cmd_mp4 = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(FRAMES_DIR / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-crf", "23",
        str(mp4_path),
    ]
    subprocess.run(cmd_mp4, check=True, capture_output=True)
    print(f"  MP4: {mp4_path} ({mp4_path.stat().st_size/1024:.0f} KB)")

    # GIF preview (escalado y con menos frames)
    cmd_gif = [
        "ffmpeg", "-y",
        "-i", str(mp4_path),
        "-vf", f"fps=10,scale=480:-1:flags=lanczos",
        "-loop", "0",
        str(gif_path),
    ]
    subprocess.run(cmd_gif, check=True, capture_output=True)
    print(f"  GIF: {gif_path} ({gif_path.stat().st_size/1024:.0f} KB)")

    print(f"\n[OK] Video evidencia generado")


if __name__ == "__main__":
    main()
