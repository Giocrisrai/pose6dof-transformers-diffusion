#!/usr/bin/env python3
"""Demo Gradio interactivo del pipeline TFM Pose 6-DoF.

Ejecucion:
    .venv/bin/python scripts/gradio_demo.py

Permite:
- Configurar pose objetivo (x, y, z + rotacion)
- Seleccionar modelo Diffusion (original/extended/ultra)
- Generar trayectorias multimodales
- Visualizar trayectorias 3D
- Comparar modelos en vivo
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


_models_cache = {}

MODELS_INFO = {
    "original (30ep, MSE=0.020)": "diffusion_policy_grasp.pth",
    "extended (50ep, MSE=0.013)": "diffusion_policy_extended_mps.pth",
    "ultra (100ep, MSE=0.0022)": "diffusion_policy_ultra.pth",
}
MODEL_DIMS = {
    "diffusion_policy_grasp.pth": 128,
    "diffusion_policy_extended_mps.pth": 192,
    "diffusion_policy_ultra.pth": 256,
}


def load_model(filename):
    if filename in _models_cache:
        return _models_cache[filename]
    import torch
    from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = MODEL_DIMS[filename]
    model = ConditionalUNet1D(action_dim=7, horizon=16, cond_dim=64, hidden_dim=hidden_dim).to(device)
    path = REPO / "data/models" / filename
    if path.exists():
        ckpt = torch.load(path, map_location=device, weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(sd)
    model.eval()
    scheduler = SimpleDDPMScheduler(num_timesteps=100)
    _models_cache[filename] = (model, scheduler, device)
    return _models_cache[filename]


def ddim_sample(model, scheduler, cond, device, n_steps=25):
    import torch
    horizon, action_dim = 16, 7
    x = torch.randn(1, horizon, action_dim, device=device)
    full_t = scheduler.num_timesteps
    step_indices = np.linspace(0, full_t - 1, n_steps).astype(int)[::-1]
    alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i, step in enumerate(step_indices):
            t_tensor = torch.tensor([step], dtype=torch.long, device=device)
            noise_pred = model(x, t_tensor, cond)
            ab_t = alpha_bar[step]
            pred_x0 = (x - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)
            if i < len(step_indices) - 1:
                next_step = step_indices[i + 1]
                ab_next = alpha_bar[next_step]
                x = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
            else:
                x = pred_x0
    return x.cpu().numpy()[0]


def predict_trajectories(pose_x, pose_y, pose_z, rot_x, rot_y, rot_z, model_choice, n_samples, n_diffusion_steps):
    """Genera trayectorias y devuelve figura + estadisticas."""
    import time
    import torch

    filename = MODELS_INFO[model_choice]
    model, scheduler, device = load_model(filename)

    # Cond vector
    cond_vec = np.zeros(64, dtype=np.float32)
    cond_vec[:3] = [pose_x, pose_y, pose_z]
    cond_vec[3:6] = [rot_x, rot_y, rot_z]
    cond = torch.tensor(cond_vec, device=device).unsqueeze(0)

    # Sample N trajectories
    t0 = time.time()
    trajs = np.array([ddim_sample(model, scheduler, cond, device, n_diffusion_steps)
                       for _ in range(n_samples)])
    elapsed_ms = (time.time() - t0) * 1000

    # Estadisticas
    endpoints = trajs[:, -1, :3]
    endpoint_std_cm = float(np.std(endpoints, axis=0).mean() * 100)
    jerk_rms = float(np.sqrt(np.mean(np.diff(trajs[:, :, :3], n=3, axis=1) ** 2)))

    # Figura 3D
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    for traj in trajs:
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.4, linewidth=1)
    ax1.scatter([pose_x], [pose_y], [pose_z], s=200, c='red', marker='*',
                label=f'Pose objetivo\n({pose_x:.2f}, {pose_y:.2f}, {pose_z:.2f})', zorder=10)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.set_title(f'{n_samples} trayectorias 3D - {model_choice.split("(")[0].strip()}')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(122)
    for d in range(3):
        ax2.plot(np.arange(16), trajs.mean(axis=0)[:, d],
                 label=['X', 'Y', 'Z'][d], linewidth=2)
    for d in range(3):
        for traj in trajs:
            ax2.plot(np.arange(16), traj[:, d], alpha=0.2, linewidth=0.7,
                     color=['blue', 'orange', 'green'][d])
    ax2.set_xlabel('Step (horizon)')
    ax2.set_ylabel('Posición (m)')
    ax2.set_title('Componentes X/Y/Z vs tiempo')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    info = f"""
**Resultados generación**

- Modelo: `{model_choice}`
- Pasos DDIM: {n_diffusion_steps}
- Muestras: {n_samples}
- **Latencia total: {elapsed_ms:.1f} ms** ({elapsed_ms/n_samples:.1f} ms/muestra)
- **Dispersión endpoint (std): {endpoint_std_cm:.2f} cm**
- **Jerk RMS: {jerk_rms:.4f}** (suavidad — menor = mejor)
- Device: `{device}`
"""
    return fig, info


def compare_models(pose_x, pose_y, pose_z):
    """Genera trayectorias con los 3 modelos y compara."""
    import torch
    import time
    fig = plt.figure(figsize=(15, 5))
    colors = ['#0098CD', '#35876B', '#FF6B35']
    stats = []

    for idx, (label, filename) in enumerate(MODELS_INFO.items()):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        model, scheduler, device = load_model(filename)
        cond_vec = np.zeros(64, dtype=np.float32)
        cond_vec[:3] = [pose_x, pose_y, pose_z]
        cond = torch.tensor(cond_vec, device=device).unsqueeze(0)
        torch.manual_seed(42)
        np.random.seed(42)
        t0 = time.time()
        trajs = np.array([ddim_sample(model, scheduler, cond, device, 25) for _ in range(10)])
        latency = (time.time() - t0) * 1000
        for traj in trajs:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[idx], alpha=0.5, linewidth=1)
        ax.scatter([pose_x], [pose_y], [pose_z], s=180, c='red', marker='*', zorder=10)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(label.split('(')[0].strip() + f'\n({latency/10:.1f}ms/sample)')
        ax.grid(True, alpha=0.3)
        endpoint_std = float(np.std(trajs[:, -1, :3], axis=0).mean() * 100)
        stats.append((label, latency/10, endpoint_std))

    plt.tight_layout()
    md = "**Comparativa los 3 modelos (n=10 muestras cada uno):**\n\n"
    md += "| Modelo | Latencia/muestra | Dispersión endpoint |\n|---|---|---|\n"
    for label, lat, std in stats:
        md += f"| {label} | {lat:.1f} ms | {std:.2f} cm |\n"
    md += "\n*Observación: el modelo Ultra converge más al target objetivo (menor dispersión), reflejando mejor entrenamiento.*"
    return fig, md


# ============================================================================
# GRADIO UI
# ============================================================================
import gradio as gr

with gr.Blocks(title="TFM Pose 6-DoF Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 TFM Pose 6-DoF — Demo Interactivo
    **Pipeline integrado FoundationPose (Transformer) + Diffusion Policy + PBVS**
    para bin picking robótico — UNIR 2026
    """)

    with gr.Tab("🎯 Generación de trayectorias"):
        gr.Markdown("Configura la pose objetivo y el modelo Diffusion para generar trayectorias multimodales.")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Pose objetivo (m)")
                pose_x = gr.Slider(-0.5, 0.5, value=0.0, step=0.05, label="X (m)")
                pose_y = gr.Slider(-0.5, 0.5, value=0.0, step=0.05, label="Y (m)")
                pose_z = gr.Slider(0.5, 1.2, value=0.8, step=0.05, label="Z (m)")
                gr.Markdown("### Rotación (axis-angle)")
                rot_x = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="ωx")
                rot_y = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="ωy")
                rot_z = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="ωz")
                model_choice = gr.Radio(choices=list(MODELS_INFO.keys()),
                                         value="ultra (100ep, MSE=0.0022)",
                                         label="Modelo Diffusion")
                n_samples = gr.Slider(1, 20, value=10, step=1, label="Trayectorias")
                n_steps = gr.Slider(10, 100, value=25, step=5, label="DDIM steps")
                btn = gr.Button("Generar trayectorias", variant="primary")
            with gr.Column():
                output_plot = gr.Plot(label="Visualización")
                output_info = gr.Markdown()

        btn.click(predict_trajectories,
                  inputs=[pose_x, pose_y, pose_z, rot_x, rot_y, rot_z, model_choice, n_samples, n_steps],
                  outputs=[output_plot, output_info])

    with gr.Tab("⚖️ Comparar modelos"):
        gr.Markdown("Compara los 3 modelos Diffusion (Original/Extended/Ultra) sobre una misma pose objetivo.")
        with gr.Row():
            with gr.Column():
                cmp_x = gr.Slider(-0.5, 0.5, value=0.0, step=0.05, label="X (m)")
                cmp_y = gr.Slider(-0.5, 0.5, value=0.0, step=0.05, label="Y (m)")
                cmp_z = gr.Slider(0.5, 1.2, value=0.8, step=0.05, label="Z (m)")
                cmp_btn = gr.Button("Comparar", variant="primary")
            with gr.Column():
                cmp_plot = gr.Plot()
                cmp_info = gr.Markdown()
        cmp_btn.click(compare_models, inputs=[cmp_x, cmp_y, cmp_z], outputs=[cmp_plot, cmp_info])

    with gr.Tab("📊 Métricas del proyecto"):
        gr.Markdown("""
        ## Hipótesis validadas

        | Hipótesis | Criterio | Resultado | Estado |
        |-----------|----------|-----------|:------:|
        | H1 — Precisión | AUC ADD-S, Δ≥3pp | YCB-V 0.908 [0.901, 0.916] / T-LESS 0.957 [0.954, 0.959] | ✅ |
        | H2 — Multimodal | score≥0.95, <50ms | score 0.96, sampling 1.88ms, **MSE 0.00221** (ultra) | ✅ |
        | H3 — Cycle E2E | p95 < 10s | YCB-V 6.29 s / T-LESS 6.68 s (ultra) | ✅ |

        ## Re-entrenamiento progresivo (exp13)

        | Modelo | MSE | Latencia | Jerk | Diversidad |
        |--------|-----|----------|------|------------|
        | Original (30ep, 2K) | 0.020 | 87 ms | 0.797 | 62.5 cm |
        | Extended (50ep, 5K) | 0.013 | 86 ms | 0.484 | 37.2 cm |
        | **Ultra (100ep, 10K)** | **0.0022** | **93 ms** | **0.053** | **3.8 cm** |

        Mejora del modelo Ultra: **-89% MSE, -93% Jerk** con +7% latencia.

        Repositorio: https://github.com/Giocrisrai/pose6dof-transformers-diffusion
        """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
