#!/usr/bin/env python3
"""Experimento 23: instrucciones SECUENCIALES multi-step.

Sistema que recibe instrucciones tipo:
- "first the red cube, then the blue sphere"
- "pick the green cylinder, then the red box"
- "grab the small box, after the large sphere"
- "in order: red sphere, blue cube, green cylinder"

Parser simple basado en regex + conectores ("first/then/after/y luego/...").
Cada sub-instruccion se ejecuta secuencialmente usando el modelo multi-objeto
del exp 20 (diffusion_policy_clip_multi.pth).

No requiere re-entrenar nada: aprovecha que el modelo del exp 20 ya selecciona
1 objeto entre N por instruccion. Solo anadimos el orquestador secuencial.

Evaluacion: % de instrucciones secuenciales donde TODOS los pasos se ejecutan
correctamente en el orden indicado.

Salida:
    experiments/results/exp23_sequential/exp23_results.json
    experiments/results/exp23_sequential/scene_*.png
"""
from __future__ import annotations
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler

HORIZON = 16
MAX_OBJ = 5
ATTR_DIM = 7
COND_DIM = 64
COLORS = ["red", "blue", "green"]
SHAPES = ["cube", "sphere", "cylinder", "box"]
COLORS_RGB_VEC = {"red": [1.0, 0.0, 0.0], "blue": [0.0, 0.0, 1.0], "green": [0.0, 1.0, 0.0]}
COLORS_RGB_HEX = {"red": "#E63946", "blue": "#0098CD", "green": "#2A9D8F"}
SHAPE_ENC = {s: [1.0 if s == x else 0.0 for x in SHAPES] for s in SHAPES}

OUTPUT = REPO / "experiments/results/exp23_sequential"
OUTPUT.mkdir(parents=True, exist_ok=True)


# Connectors que separan sub-instrucciones (en ingles y espanol)
CONNECTORS = [
    r",\s*then\s+",
    r",\s*after\s+",
    r",\s*next\s+",
    r"\s+then\s+",
    r"\s+after\s+",
    r"\s+and\s+then\s+",
    r"\s+followed\s+by\s+",
]
CONNECTOR_RE = re.compile("|".join(CONNECTORS), flags=re.IGNORECASE)
PREFIX_RE = re.compile(r"^(first|in order:|sequence:|please)\s+", flags=re.IGNORECASE)


def parse_sequence(text: str) -> list[str]:
    """Parsea "first the red cube, then the blue sphere" -> ['the red cube', 'the blue sphere'].

    Estrategia: quitar prefijo (first/in order/sequence/please) + split por connectors.
    Si no hay connectors, devuelve [text] (instruccion simple).
    """
    text = text.strip()
    text = PREFIX_RE.sub("", text)
    # Caso "in order: a, b, c" -> dividir por comas
    if re.search(r"[,;]", text) and not CONNECTOR_RE.search(text):
        parts = [p.strip() for p in re.split(r"[,;]", text) if p.strip()]
        if len(parts) >= 2:
            return parts
    # Split por connectors
    parts = CONNECTOR_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


# ============================================================================
# MODELO (reutiliza el del exp 20)
# ============================================================================
class _Proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 64), nn.Mish(), nn.Linear(64, 32))
    def forward(self, x): return self.net(x)


class _Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(7+512, 128), nn.Mish(),
                                      nn.Linear(128, 128), nn.Mish(),
                                      nn.Linear(128, 1))
    def forward(self, ce, attrs, mask):
        B, N, _ = attrs.shape
        x = torch.cat([attrs, ce.unsqueeze(1).expand(-1, N, -1)], -1)
        logits = self.score(x).squeeze(-1).masked_fill(mask == 0, -1e9)
        return F.softmax(logits, dim=-1)


@dataclass
class StepResult:
    sub_text: str
    chosen_idx: int
    confidence: float
    distance_to_target_cm: float
    trajectory: np.ndarray


def load_model():
    from transformers import CLIPTokenizer, CLIPTextModel
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_mod = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    ckpt = torch.load(REPO / "data/models/diffusion_policy_clip_multi.pth",
                       map_location=device, weights_only=True)
    model = ConditionalUNet1D(action_dim=7, horizon=16, cond_dim=64, hidden_dim=256).to(device).eval()
    proj = _Proj().to(device).eval()
    gate = _Gate().to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    proj.load_state_dict(ckpt["projector_state_dict"])
    gate.load_state_dict(ckpt["gate_state_dict"])
    return tok, clip_mod, model, proj, gate, device


def run_single_step(sub_text: str, scene_objects, tok, clip_mod, model, proj, gate, device):
    """Ejecuta una sub-instruccion sobre los objetos disponibles."""
    n = len(scene_objects)
    attrs = np.zeros((MAX_OBJ, ATTR_DIM), dtype=np.float32)
    positions = np.zeros((MAX_OBJ, 3), dtype=np.float32)
    mask = np.zeros(MAX_OBJ, dtype=np.float32)
    for i, obj in enumerate(scene_objects):
        attrs[i, :3] = COLORS_RGB_VEC[obj["color"]]
        attrs[i, 3:7] = SHAPE_ENC[obj["shape"]]
        positions[i] = obj["pos"]
        mask[i] = 1.0

    with torch.no_grad():
        ins = tok([sub_text], padding=True, return_tensors="pt", truncation=True)
        ins = {k: v.to(device) for k, v in ins.items()}
        ce = clip_mod(**ins).pooler_output
        at = torch.tensor(attrs, device=device, dtype=torch.float32).unsqueeze(0)
        pt = torch.tensor(positions, device=device, dtype=torch.float32).unsqueeze(0)
        mt = torch.tensor(mask, device=device, dtype=torch.float32).unsqueeze(0)
        gates = gate(ce, at, mt)
        selected = (gates.unsqueeze(-1) * pt).sum(dim=1)
        clip_proj = proj(ce)
        cb = torch.zeros(1, COND_DIM, device=device, dtype=torch.float32)
        cb[0, :3] = selected[0]
        cb[0, 3:18] = pt[0].flatten()[:15]
        cb[0, 18:50] = clip_proj[0]
        cb[0, 50:50+ATTR_DIM] = at[0].max(dim=0).values
        # DDIM
        scheduler = SimpleDDPMScheduler(num_timesteps=100)
        x = torch.randn(1, HORIZON, 7, device=device, dtype=torch.float32)
        si = np.linspace(0, 99, 25).astype(int)[::-1]
        ab = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
        for i, step in enumerate(si):
            t = torch.full((1,), int(step), dtype=torch.long, device=device)
            e = model(x, t, cb)
            ab_t = ab[step]
            x0 = (x - torch.sqrt(1 - ab_t) * e) / torch.sqrt(ab_t)
            if i < len(si) - 1:
                ab_n = ab[si[i+1]]
                x = torch.sqrt(ab_n) * x0 + torch.sqrt(1 - ab_n) * e
            else:
                x = x0
        traj = x.cpu().numpy()[0]
        gates_np = gates.cpu().numpy()[0]
    chosen = int(np.argmax(gates_np[:n]))
    return chosen, float(gates_np[chosen]), traj


def execute_sequence(text: str, initial_objects: list, target_sequence: list,
                       models, with_removal: bool = True):
    """Ejecuta una instruccion secuencial.

    initial_objects: lista de dicts {color, shape, pos}.
    target_sequence: lista de indices target esperados (en el orden original).
    with_removal: si True, el objeto seleccionado se quita de la escena (mas realista).

    Devuelve: lista de StepResult + flag step_correct + overall_correct.
    """
    tok, clip_mod, model, proj, gate, device = models
    parts = parse_sequence(text)
    objects = [dict(o) for o in initial_objects]  # copy
    # Indices del objeto en la escena ORIGINAL (para mapear de vuelta)
    original_indices = list(range(len(initial_objects)))

    step_results = []
    step_correctness = []
    for k, sub in enumerate(parts):
        if len(objects) == 0:
            break
        chosen, conf, traj = run_single_step(sub, objects, tok, clip_mod, model, proj, gate, device)
        # Mapear al indice original
        chosen_original = original_indices[chosen]
        expected_original = target_sequence[k] if k < len(target_sequence) else -1
        endpoint = traj[-1, :3]
        target_pos = initial_objects[expected_original]["pos"] if expected_original >= 0 else None
        d = float(np.linalg.norm(endpoint - target_pos) * 100) if target_pos is not None else -1
        correct = chosen_original == expected_original
        step_correctness.append(correct)
        step_results.append(StepResult(
            sub_text=sub, chosen_idx=chosen_original, confidence=conf,
            distance_to_target_cm=d, trajectory=traj,
        ))
        # Remove chosen object si configurado
        if with_removal:
            del objects[chosen]
            del original_indices[chosen]

    return {
        "parsed_steps": parts,
        "step_results": step_results,
        "step_correctness": step_correctness,
        "overall_correct": all(step_correctness),
        "n_expected": len(target_sequence),
        "n_executed": len(step_results),
    }


# ============================================================================
# DEMO SCENES
# ============================================================================
def make_scene(text, objects_def, target_sequence):
    return {
        "text": text,
        "objects": [{"color": c, "shape": s, "pos": np.array(p)} for c, s, p in objects_def],
        "target_sequence": target_sequence,  # lista de indices originales
    }


DEMO_SCENES = [
    make_scene("first the red cube, then the blue sphere",
                [("red", "cube", [-0.30, 0.10, 0.80]),
                 ("blue", "sphere", [0.20, -0.10, 0.85]),
                 ("green", "cylinder", [0.35, 0.20, 0.78])], [0, 1]),
    make_scene("pick the green cylinder, then the red box",
                [("blue", "sphere", [-0.35, 0.15, 0.82]),
                 ("green", "cylinder", [0.00, -0.15, 0.85]),
                 ("red", "box", [0.30, 0.10, 0.78])], [1, 2]),
    make_scene("grab the blue cube and then the green sphere",
                [("blue", "cube", [-0.30, -0.10, 0.85]),
                 ("red", "cylinder", [0.00, 0.20, 0.80]),
                 ("green", "sphere", [0.30, -0.15, 0.78])], [0, 2]),
    make_scene("first the green box, then the red sphere, then the blue cylinder",
                [("green", "box", [-0.35, 0.10, 0.80]),
                 ("blue", "cylinder", [0.10, 0.20, 0.82]),
                 ("red", "sphere", [0.30, -0.10, 0.85])], [0, 2, 1]),
    make_scene("in order: red cube, blue box, green sphere",
                [("blue", "box", [-0.30, -0.10, 0.85]),
                 ("red", "cube", [0.00, 0.15, 0.80]),
                 ("green", "sphere", [0.30, -0.05, 0.82]),
                 ("blue", "cylinder", [0.35, 0.20, 0.78])], [1, 0, 2]),
    make_scene("take the red sphere followed by the blue cube",
                [("red", "sphere", [-0.30, 0.10, 0.82]),
                 ("blue", "cube", [0.00, -0.15, 0.85]),
                 ("green", "cylinder", [0.30, 0.10, 0.80])], [0, 1]),
    make_scene("first the blue sphere then the red cylinder then the green box then the red sphere",
                [("blue", "sphere", [-0.40, 0.10, 0.82]),
                 ("red", "cylinder", [-0.10, -0.15, 0.85]),
                 ("green", "box", [0.20, 0.20, 0.80]),
                 ("red", "sphere", [0.40, 0.00, 0.78])], [0, 1, 2, 3]),
    make_scene("pick the small green cube, then the large red sphere",
                [("green", "cube", [-0.30, 0.10, 0.82]),
                 ("red", "sphere", [0.10, -0.15, 0.85]),
                 ("blue", "cylinder", [0.30, 0.15, 0.78])], [0, 1]),
]


# ============================================================================
# RENDER
# ============================================================================
def draw_object(ax, center, color, shape):
    c = COLORS_RGB_HEX[color]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cx, cy, cz = center
    if shape == "cube":
        r = 0.025
        v = [[cx-r,cy-r,cz-r],[cx+r,cy-r,cz-r],[cx+r,cy+r,cz-r],[cx-r,cy+r,cz-r],
             [cx-r,cy-r,cz+r],[cx+r,cy-r,cz+r],[cx+r,cy+r,cz+r],[cx-r,cy+r,cz+r]]
        faces = [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],[v[0],v[1],v[5],v[4]],
                 [v[2],v[3],v[7],v[6]],[v[0],v[3],v[7],v[4]],[v[1],v[2],v[6],v[5]]]
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.5, facecolor=c, edgecolor=c))
    elif shape == "sphere":
        u, v_ = np.linspace(0, 2*np.pi, 18), np.linspace(0, np.pi, 12)
        x = cx + 0.03 * np.outer(np.cos(u), np.sin(v_))
        y = cy + 0.03 * np.outer(np.sin(u), np.sin(v_))
        z = cz + 0.03 * np.outer(np.ones_like(u), np.cos(v_))
        ax.plot_surface(x, y, z, color=c, alpha=0.6, linewidth=0.2)
    elif shape == "cylinder":
        th = np.linspace(0, 2*np.pi, 22); z = np.linspace(cz-0.035, cz+0.035, 8)
        T, Z = np.meshgrid(th, z)
        ax.plot_surface(cx + 0.025*np.cos(T), cy + 0.025*np.sin(T), Z,
                          color=c, alpha=0.55, linewidth=0.2)
    elif shape == "box":
        sx, sy, sz = 0.035, 0.022, 0.018
        v = [[cx-sx,cy-sy,cz-sz],[cx+sx,cy-sy,cz-sz],[cx+sx,cy+sy,cz-sz],[cx-sx,cy+sy,cz-sz],
             [cx-sx,cy-sy,cz+sz],[cx+sx,cy-sy,cz+sz],[cx+sx,cy+sy,cz+sz],[cx-sx,cy+sy,cz+sz]]
        faces = [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],[v[0],v[1],v[5],v[4]],
                 [v[2],v[3],v[7],v[6]],[v[0],v[3],v[7],v[4]],[v[1],v[2],v[6],v[5]]]
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.5, facecolor=c, edgecolor=c))


def render_sequence(scene_idx, scene, result, save_path):
    """Renderiza la ejecucion secuencial: escena 3D con trayectorias coloreadas por paso."""
    fig = plt.figure(figsize=(13, 6.5))
    ax = fig.add_subplot(121, projection="3d")

    xs = np.linspace(-0.5, 0.5, 2); ys = np.linspace(-0.5, 0.5, 2)
    xx, yy = np.meshgrid(xs, ys)
    ax.plot_surface(xx, yy, np.full_like(xx, 0.7), alpha=0.08, color="gray")

    # Objetos con etiquetas
    for i, obj in enumerate(scene["objects"]):
        draw_object(ax, obj["pos"], obj["color"], obj["shape"])
        ax.text(obj["pos"][0], obj["pos"][1], obj["pos"][2] + 0.07,
                  f"#{i+1}\n{obj['color']}\n{obj['shape']}",
                  ha="center", fontsize=8)

    # Trayectorias con colormap por orden
    cmap = plt.cm.plasma
    n_steps = len(result["step_results"])
    for k, step in enumerate(result["step_results"]):
        color = cmap(0.15 + 0.7 * k / max(n_steps - 1, 1))
        traj = step.trajectory
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=2.5, alpha=0.9,
                  label=f"Paso {k+1}: '{step.sub_text[:25]}...' -> #{step.chosen_idx+1}")
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]], s=120,
                       c=[color], marker="*", edgecolor="white", linewidth=1)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0.65, 1.1)
    ax.set_title(f"Secuencia #{scene_idx+1}", fontsize=11)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.2)

    # Texto explicativo
    ax2 = fig.add_subplot(122)
    ax2.set_axis_off()
    text_lines = [
        f"Instruccion completa:",
        f'   "{scene["text"]}"',
        "",
        f"Sub-pasos parseados ({len(result['parsed_steps'])}):"
    ]
    for k, sub in enumerate(result["parsed_steps"]):
        text_lines.append(f"   {k+1}. \"{sub}\"")
    text_lines.append("")
    text_lines.append("Resultado por paso:")
    for k, (step, ok) in enumerate(zip(result["step_results"], result["step_correctness"])):
        obj = scene["objects"][step.chosen_idx]
        expected = scene["target_sequence"][k] if k < len(scene["target_sequence"]) else -1
        exp_obj = scene["objects"][expected] if expected >= 0 else None
        flag = "✓" if ok else "✗"
        text_lines.append(f"   {flag} Paso {k+1}: eligio #{step.chosen_idx+1} "
                            f"({obj['color']} {obj['shape']}) "
                            f"conf={step.confidence:.0%}")
        if exp_obj:
            text_lines.append(f"      esperaba #{expected+1} ({exp_obj['color']} {exp_obj['shape']})")
        text_lines.append(f"      distancia: {step.distance_to_target_cm:.1f} cm")
    text_lines.append("")
    flag_overall = "✓ TODOS CORRECTOS" if result["overall_correct"] else "✗ ALGUN PASO FALLO"
    text_lines.append(f"OVERALL: {flag_overall}")

    bg_color = "#f0fdf4" if result["overall_correct"] else "#fee2e2"
    edge_color = "#16a34a" if result["overall_correct"] else "#dc2626"
    ax2.text(0.02, 0.98, "\n".join(text_lines), transform=ax2.transAxes, fontsize=9,
              verticalalignment="top", family="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color, edgecolor=edge_color))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    print("[exp23] Ejecucion de instrucciones secuenciales multi-step")
    print(f"  Reusando modelo del exp 20 (diffusion_policy_clip_multi.pth)")
    print(f"  {len(DEMO_SCENES)} escenas demostrativas")
    models = load_model()
    print(f"  device={models[-2]}")

    results = {"scenes": []}
    correct = 0
    total_steps = 0
    correct_steps = 0
    for i, scene in enumerate(DEMO_SCENES):
        r = execute_sequence(scene["text"], scene["objects"], scene["target_sequence"],
                              models, with_removal=True)
        if r["overall_correct"]:
            correct += 1
        for ok in r["step_correctness"]:
            total_steps += 1
            if ok: correct_steps += 1
        flag = "✓" if r["overall_correct"] else "✗"
        print(f"  {flag} #{i+1}: '{scene['text'][:60]}...' | "
              f"steps {sum(r['step_correctness'])}/{len(r['step_results'])}")
        save_path = OUTPUT / f"scene_{i+1:02d}.png"
        render_sequence(i, scene, r, save_path)
        results["scenes"].append({
            "id": i+1, "text": scene["text"],
            "parsed_steps": r["parsed_steps"], "step_correctness": r["step_correctness"],
            "overall_correct": r["overall_correct"],
            "step_details": [
                {"sub_text": s.sub_text, "chosen_idx": s.chosen_idx,
                  "confidence": s.confidence, "distance_cm": s.distance_to_target_cm}
                for s in r["step_results"]],
        })

    print(f"\n  Overall: {correct}/{len(DEMO_SCENES)} secuencias completas ({correct/len(DEMO_SCENES):.1%})")
    print(f"  Step-level: {correct_steps}/{total_steps} = {correct_steps/total_steps:.1%}")
    results["summary"] = {
        "n_scenes": len(DEMO_SCENES), "n_correct_sequences": correct,
        "sequence_accuracy": correct / len(DEMO_SCENES),
        "n_steps_total": total_steps, "n_correct_steps": correct_steps,
        "step_accuracy": correct_steps / total_steps,
    }
    with open(OUTPUT / "exp23_results.json", "w") as f:
        # Convertir np types
        def conv(o):
            if isinstance(o, (np.bool_, np.integer)): return int(o) if not isinstance(o, np.bool_) else bool(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
        json.dump(results, f, indent=2, default=conv)
    print(f"\n[OK] {OUTPUT / 'exp23_results.json'}")

    # Grid overview
    rows = (len(DEMO_SCENES) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(20, 6 * rows))
    for ax, i in zip(axes.flat, range(len(DEMO_SCENES))):
        from matplotlib.image import imread
        img = imread(OUTPUT / f"scene_{i+1:02d}.png")
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"#{i+1}: {DEMO_SCENES[i]['text'][:50]}...", fontsize=9)
    plt.suptitle(f"Instrucciones secuenciales — {correct}/{len(DEMO_SCENES)} secuencias completas",
                  fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT / "grid_overview.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] {OUTPUT / 'grid_overview.png'}")


if __name__ == "__main__":
    main()
