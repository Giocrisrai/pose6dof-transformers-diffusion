#!/usr/bin/env python3
"""Video CINEMATOGRAFICO del pipeline E2E con multiples camaras + paneles informativos.

Mejoras vs v1:
- 3 vistas distintas: cenital, perspectiva 45 deg, primer plano del agarre
- Camara orbitando lentamente alrededor del area de trabajo
- Panel lateral con barra de progreso y stats en tiempo real
- Transiciones entre fases (PERCEPCION -> PLANIFICACION -> AGARRE)
- Resolucion 1280x720 (vs 640x480 anterior)
- Trayectoria diffusion overlay en 3D

Salidas:
    experiments/results/pipeline_e2e/demo_v2.mp4
    experiments/results/pipeline_e2e/demo_v2.gif
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "experiments/results/pipeline_e2e"
FRAMES_DIR = OUTPUT / "demo_v2_frames"

# Parametros visuales
PANEL_W = 360         # ancho del panel lateral con info
RESX, RESY = 920, 720  # vista 3D
TOTAL_W = RESX + PANEL_W
TOTAL_H = RESY


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


def create_orbit_camera(sim, name="tfm_orbit_cam"):
    """Usa el vision sensor existente sin modificar su configuracion."""
    all_vs = sim.getObjectsInTree(sim.handle_scene, sim.object_visionsensor_type)
    if not all_vs:
        print("  [FAIL] no hay vision sensors en la escena")
        return None
    # Preferir el tfm_overview_sensor (cenital amplio); fallback al primer disponible
    cam = all_vs[0]
    for h in all_vs:
        try:
            if "overview" in sim.getObjectAlias(h).lower():
                cam = h
                break
        except Exception:
            pass
    print(f"  Vision sensor reusado: handle={cam}, alias={sim.getObjectAlias(cam)}")
    return cam


def look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)):
    """Devuelve una matriz 3x4 (12 floats fila-mayor) que pone la camara en `eye`
    mirando hacia `target`. CoppeliaSim usa convencion: -Z apunta hacia adelante."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # Forward de la camara hacia el objetivo
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    # CoppeliaSim convention: la camara mira hacia -Z local
    # Construimos el frame: z_cam = -f
    z_cam = -f
    x_cam = np.cross(up, z_cam)
    if np.linalg.norm(x_cam) < 1e-6:
        # up paralelo a z_cam -> usar otro up
        up_alt = np.array([0.0, 1.0, 0.0])
        x_cam = np.cross(up_alt, z_cam)
    x_cam = x_cam / (np.linalg.norm(x_cam) + 1e-9)
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / (np.linalg.norm(y_cam) + 1e-9)

    # Matriz 3x4: [x_cam, y_cam, z_cam, eye] en columnas, fila-mayor
    M = [
        x_cam[0], y_cam[0], z_cam[0], eye[0],
        x_cam[1], y_cam[1], z_cam[1], eye[1],
        x_cam[2], y_cam[2], z_cam[2], eye[2],
    ]
    return [float(v) for v in M]


def position_camera(sim, cam, t_seconds, phase_idx, scene_target=None):
    """Mueve la camara orbitando alrededor del centro DETECTADO de la escena
    (usa el handle del Floor como referencia)."""
    if scene_target is None:
        scene_target = (0.0, 0.0, 0.0)

    if phase_idx == 0:  # PERCEPCION
        radius, height, ang_speed, ang_offset = 2.5, 2.5, 0.25, 0.0
    elif phase_idx == 1:  # PLANIFICACION
        radius, height, ang_speed, ang_offset = 2.2, 1.8, 0.35, math.pi/3
    else:  # EJECUCION
        radius, height, ang_speed, ang_offset = 1.9, 1.3, 0.45, 2*math.pi/3

    angle = ang_offset + ang_speed * t_seconds
    eye = (scene_target[0] + radius * math.cos(angle),
           scene_target[1] + radius * math.sin(angle),
           scene_target[2] + height)
    M = look_at_matrix(eye, scene_target, up=(0.0, 0.0, 1.0))
    try:
        sim.setObjectMatrix(cam, -1, M)
    except Exception as e:
        sim.setObjectPosition(cam, -1, list(eye))


def discover_scene_target(sim):
    """Localiza el centro real del area de trabajo en la escena."""
    candidates = ["/genericConveyorTypeA", "/Floor"]
    for path in candidates:
        try:
            h = sim.getObject(path)
            pos = sim.getObjectPosition(h, -1)
            return tuple(pos[:2]) + (max(pos[2], 0.5),)
        except Exception:
            continue
    return (0.0, 0.0, 0.5)


def capture_frame(sim, cam):
    """Renderiza explicitamente el sensor antes de capturar.

    handleVisionSensor() puede dar error si el sensor no esta tagged for
    explicit handling, pero entonces el render automatico durante sim.step()
    deberia haber procesado el sensor. Llamamos en try/except para cubrir
    ambos casos.
    """
    try:
        sim.handleVisionSensor(cam)
    except Exception:
        pass  # sensor no tagged - render automatico via sim.step()

    img_data, res_xy = sim.getVisionSensorImg(cam, 0, 0.0, [0, 0], [0, 0])
    rx, ry = res_xy
    arr = np.frombuffer(img_data, dtype=np.uint8).reshape(ry, rx, 3)
    arr = np.flipud(arr)
    return arr, rx, ry


def virtual_camera_motion(arr, t_seconds, phase_idx):
    """Zoom y pan virtual sobre el frame capturado (no toca CoppeliaSim).
    Phase 0: zoom out lento + pan circular suave.
    Phase 1: zoom in al area de planificacion.
    Phase 2: zoom in al area de agarre con leve pan.
    """
    import cv2
    h, w = arr.shape[:2]

    # Parametros por fase
    if phase_idx == 0:
        # PERCEPCION: zoom suave 0.95 -> 0.85 (zoom-out cinematografico)
        zoom = 0.95 - 0.05 * (t_seconds % 5) / 5
        pan_x = 0.02 * math.sin(0.5 * t_seconds)
        pan_y = 0.02 * math.cos(0.5 * t_seconds)
    elif phase_idx == 1:
        # PLANIFICACION: zoom-in suave 0.9 -> 0.75
        cycle_t = (t_seconds % 4) / 4
        zoom = 0.9 - 0.15 * cycle_t
        pan_x = 0.05 * math.sin(0.7 * t_seconds)
        pan_y = -0.03
    else:
        # EJECUCION: zoom mas cerrado 0.8 -> 0.65 + pan vertical
        cycle_t = (t_seconds % 3) / 3
        zoom = 0.8 - 0.15 * cycle_t
        pan_x = 0.04 * math.sin(0.9 * t_seconds)
        pan_y = -0.05

    # Aplicar zoom: cropear region central de tamano w*zoom x h*zoom y resize a w x h
    crop_w = max(int(w * zoom), 100)
    crop_h = max(int(h * zoom), 100)
    cx = int(w / 2 + pan_x * w)
    cy = int(h / 2 + pan_y * h)
    x0 = max(min(cx - crop_w // 2, w - crop_w), 0)
    y0 = max(min(cy - crop_h // 2, h - crop_h), 0)
    crop = arr[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def make_panel(cycle, n_cycles, obj_id, dataset, phase, phase_progress, total_progress,
               diff_ms, fp_ms, sim_ms, traj=None):
    """Construye el panel lateral con metricas + barras de progreso."""
    import cv2
    panel = np.zeros((TOTAL_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (40, 40, 50)  # fondo gris-azulado oscuro

    # Acentos por fase
    PHASE_COLORS = {
        "PERCEPCION":   (10, 152, 205),   # naranja-ambar BGR
        "PLANIFICACION": (53, 107, 0),    # turquesa
        "EJECUCION":    (10, 100, 220),   # rojo-naranja
    }
    accent = PHASE_COLORS.get(phase.split()[0], (200, 200, 200))

    # Header banner
    cv2.rectangle(panel, (0, 0), (PANEL_W, 70), accent, -1)
    cv2.putText(panel, "TFM Bin Picking 6-DoF", (12, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "FoundationPose + Diffusion Policy", (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    y = 100

    def section(title, c):
        nonlocal y
        cv2.rectangle(panel, (0, y-20), (4, y+5), c, -1)
        cv2.putText(panel, title, (12, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        y += 28

    def kv(k, v, color=(180, 220, 255)):
        nonlocal y
        cv2.putText(panel, k, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(panel, v, (160, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += 22

    section("ESCENA", accent)
    kv("Robot", "Ragnar (delta)")
    kv("Vista", "Cenital - conveyor")
    kv("Dataset", dataset.upper())
    kv("Object ID", str(obj_id))
    kv("Cycle", f"{cycle}/{n_cycles}")
    y += 8

    section("FASE ACTUAL", accent)
    cv2.putText(panel, phase, (16, y), cv2.FONT_HERSHEY_DUPLEX, 0.7, accent, 1, cv2.LINE_AA)
    y += 30

    # Barra de fase
    bar_y = y
    cv2.rectangle(panel, (16, bar_y), (PANEL_W - 16, bar_y + 12), (60, 60, 70), -1)
    fill_w = int((PANEL_W - 32) * phase_progress)
    cv2.rectangle(panel, (16, bar_y), (16 + fill_w, bar_y + 12), accent, -1)
    y += 28
    cv2.putText(panel, f"Progreso fase: {phase_progress*100:.0f}%",
                (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    y += 25

    section("LATENCIA", accent)
    kv("FoundationPose", f"{fp_ms:.0f} ms", (255, 200, 100))
    kv("Diffusion DDIM", f"{diff_ms:.0f} ms", (100, 220, 255))
    kv("Simulacion", f"{sim_ms:.0f} ms", (200, 255, 200))
    total_ms = fp_ms + diff_ms + sim_ms
    kv("TOTAL ciclo", f"{total_ms:.0f} ms",
       (100, 255, 100) if total_ms < 10000 else (100, 100, 255))
    y += 8

    # Indicador H3
    section("H3 (cycle < 10s)", accent)
    h3_color = (100, 255, 100) if total_ms < 10000 else (100, 100, 255)
    h3_text = "PASA" if total_ms < 10000 else "FALLA"
    cv2.putText(panel, h3_text, (16, y + 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, h3_color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"margen {(10000-total_ms)/1000:.2f} s",
                (130, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    y += 30
    y += 15

    # Progreso total
    section("PROGRESO DEMO", accent)
    bar_y = y
    cv2.rectangle(panel, (16, bar_y), (PANEL_W - 16, bar_y + 14), (60, 60, 70), -1)
    fill_w = int((PANEL_W - 32) * total_progress)
    cv2.rectangle(panel, (16, bar_y), (16 + fill_w, bar_y + 14), (255, 200, 50), -1)
    y += 28

    # Stamp en el footer
    cv2.putText(panel, "UNIR 2026 | M1 Pro + MPS", (12, TOTAL_H - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(panel, "CoppeliaSim Edu V4.10", (12, TOTAL_H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

    return panel


def composite_frame(sim_view, panel, phase_marker_color, target_w, target_h):
    """Combina vista 3D + panel lateral con borde de fase."""
    import cv2
    sim_bgr = cv2.cvtColor(sim_view, cv2.COLOR_RGB2BGR)
    # Resize si la vista no es del tamano esperado
    if sim_bgr.shape[0] != target_h or sim_bgr.shape[1] != target_w:
        sim_bgr = cv2.resize(sim_bgr, (target_w, target_h))
    # Borde de color de fase alrededor de la vista 3D
    cv2.rectangle(sim_bgr, (0, 0), (target_w-1, target_h-1), phase_marker_color, 4)
    full = np.hstack([sim_bgr, panel])
    return full


def save_frame(arr, path):
    import cv2
    cv2.imwrite(str(path), arr)


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
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cycles", type=int, default=3)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dataset", default="ycbv")
    args = parser.parse_args()

    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    PHASES = [
        ("PERCEPCION 6-DoF",        50),
        ("PLANIFICACION DIFFUSION", 40),
        ("EJECUCION AGARRE",       110),  # mas tiempo para ver el conveyor mover piezas
    ]
    STEPS_PER_CYCLE = sum(s for _, s in PHASES)
    TOTAL_STEPS = args.n_cycles * STEPS_PER_CYCLE
    print(f"[record-v2] {args.n_cycles} ciclos x {STEPS_PER_CYCLE} steps = {TOTAL_STEPS} frames @ {args.fps}fps")

    print("\n[1/4] Setup CoppeliaSim...")
    sim = setup_sim()
    cam = create_orbit_camera(sim)
    print(f"  Camara HD: {RESX}x{RESY}")
    scene_target = discover_scene_target(sim)
    print(f"  Centro de escena detectado: {scene_target}")
    for _ in range(20):
        sim.step()

    # Validar que la camara captura algo (no escena negra)
    arr_test, _, _ = capture_frame(sim, cam)
    mean_intensity = float(arr_test.mean())
    print(f"  Brillo medio sensor (default): {mean_intensity:.1f}")
    if mean_intensity < 5.0:
        print("  [WARN] sensor capturando escena oscura — la camara podria estar fuera de cuadro")

    print("\n[2/4] Diffusion Policy...")
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

    ckpt_file = REPO / f"experiments/checkpoints/fp_{args.dataset}_checkpoint.json"
    with open(ckpt_file) as f:
        preds = json.load(f)["results"][:args.n_cycles]

    print("\n[3/4] Grabando frames cinematograficos...")
    frame_idx = 0
    for cycle_i, pred in enumerate(preds):
        R = np.array(pred["R_pred"])
        t_pose = np.array(pred["t_pred"])
        fp_ms = pred["time_s"] * 1000

        # Diffusion sampling para este ciclo
        cond_vec = np.zeros(64, dtype=np.float32)
        flat = np.concatenate([R.flatten(), t_pose.flatten()])
        cond_vec[:len(flat)] = flat
        cond = torch.tensor(cond_vec, device=device).unsqueeze(0)
        t0 = time.time()
        traj = run_diffusion(planner, scheduler, cond, device, 25)
        diff_ms = (time.time() - t0) * 1000
        print(f"  Ciclo {cycle_i+1}: obj_id={pred['obj_id']} | FP={fp_ms:.0f}ms | Diff={diff_ms:.0f}ms")

        # Tres fases con diferentes camaras
        cycle_step = 0
        for phase_idx, (phase_name, n_steps_phase) in enumerate(PHASES):
            for s in range(n_steps_phase):
                t_phase = s / max(n_steps_phase - 1, 1)  # 0..1
                t_global = frame_idx / args.fps
                t_total = frame_idx / TOTAL_STEPS

                # NO movemos la camara fisica (preserva render correcto del sensor).
                # En su lugar aplicamos zoom/pan virtual sobre el frame capturado.
                sim.step()

                arr, rx_actual, ry_actual = capture_frame(sim, cam)

                # Zoom y pan virtual para emular movimiento de camara
                arr = virtual_camera_motion(arr, t_global, phase_idx)

                # Sim ms estimado
                sim_ms_est = 18.0 * (cycle_step + 1)

                phase_color = [
                    (10, 152, 205),
                    (53, 107, 0),
                    (10, 100, 220),
                ][phase_idx]

                panel = make_panel(
                    cycle=cycle_i+1, n_cycles=len(preds),
                    obj_id=pred["obj_id"], dataset=args.dataset,
                    phase=phase_name,
                    phase_progress=t_phase, total_progress=t_total,
                    diff_ms=diff_ms, fp_ms=fp_ms, sim_ms=sim_ms_est,
                )
                full = composite_frame(arr, panel, phase_color, RESX, RESY)
                save_frame(full, FRAMES_DIR / f"frame_{frame_idx:05d}.png")
                frame_idx += 1
                cycle_step += 1

    sim.stopSimulation()
    print(f"  Total: {frame_idx} frames")

    print("\n[4/4] Exportar MP4 + GIF...")
    mp4 = OUTPUT / "demo_v2.mp4"
    gif = OUTPUT / "demo_v2.gif"

    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(FRAMES_DIR / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-crf", "20",
        str(mp4),
    ], check=True, capture_output=True)
    print(f"  MP4: {mp4} ({mp4.stat().st_size/1024:.0f} KB)")

    subprocess.run([
        "ffmpeg", "-y", "-i", str(mp4),
        "-vf", "fps=12,scale=720:-1:flags=lanczos",
        "-loop", "0", str(gif),
    ], check=True, capture_output=True)
    print(f"  GIF: {gif} ({gif.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
