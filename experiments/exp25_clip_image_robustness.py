#!/usr/bin/env python3
"""Experimento 25: robustez del modelo CLIP-image (exp 24) ante perturbaciones.

Evaluamos el modelo `diffusion_policy_clip_image.pth` con crops sometidos a
DOMAIN RANDOMIZATION agresivo, simulando las condiciones del mundo real:

1. Oclusion parcial: 0 / 20 / 40 / 60 %
2. Ruido gaussiano: sigma 0 / 10 / 25 / 50 (en intensidad 0-255)
3. Cambio de iluminacion (gain): 0.5x / 0.75x / 1.0x / 1.5x / 2.0x

Cada condicion se evalua sobre 300 escenas con bootstrap CI 95% usando
bop-bootstrap-ci (cierra el ciclo con exp 1).

Salida:
    experiments/results/exp25_robustness/exp25_results.json
    experiments/results/exp25_robustness/fig_robustness_curves.png
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bop_bootstrap_ci import bootstrap_recall  # reusa el paquete del exp 1
from experiments.exp24_clip_image_grounding import (
    CLIPEncoders, VisualGate, CLIPProjector,
    render_object_crop, generate_scene, MAX_OBJ, COLORS, SHAPES,
)

OUTPUT = REPO / "experiments/results/exp25_robustness"
OUTPUT.mkdir(parents=True, exist_ok=True)


def perturb_crop(crop: np.ndarray, occlusion_pct: float, noise_sigma: float,
                   illum_gain: float, rng) -> np.ndarray:
    """Aplica perturbaciones controladas al crop."""
    img = crop.astype(np.float32).copy()

    # 1. Iluminacion
    img = np.clip(img * illum_gain, 0, 255)

    # 2. Ruido gaussiano
    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, size=img.shape)
        img = np.clip(img + noise, 0, 255)

    # 3. Oclusion: tapar un rectangulo del % indicado del area
    if occlusion_pct > 0:
        H, W = img.shape[:2]
        area_target = occlusion_pct * H * W
        side = int(np.sqrt(area_target))
        side = min(side, min(H, W) - 2)
        x0 = rng.integers(0, max(W - side, 1))
        y0 = rng.integers(0, max(H - side, 1))
        # Tapar con color aleatorio gris
        cover_color = rng.integers(80, 180)
        img[y0:y0+side, x0:x0+side] = cover_color

    return img.astype(np.uint8)


def load_model(device):
    from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPVisionModel
    clip = CLIPEncoders(device)
    ckpt = torch.load(REPO / "data/models/diffusion_policy_clip_image.pth",
                       map_location=device, weights_only=True)
    proj = CLIPProjector(512, 32).to(device).eval()
    gate = VisualGate(text_dim=512, vis_dim=768).to(device).eval()
    proj.load_state_dict(ckpt["projector_state_dict"])
    gate.load_state_dict(ckpt["gate_state_dict"])
    return clip, proj, gate


def evaluate_condition(scenes, clip, gate, occlusion, noise, illum, device, seed=42):
    """Para cada escena, perturba los crops y evalua si el gate elige correctamente.

    Devuelve array binario per-instance (1 si correcto) para bootstrap.
    """
    rng = np.random.default_rng(seed)
    correct = np.zeros(len(scenes), dtype=np.int32)
    confidences = []

    for i, s in enumerate(scenes):
        # Render + perturb crops
        crops_pert = []
        for obj in s["objects"]:
            crop = render_object_crop(obj["color"], obj["shape"], rng=rng)
            crop = perturb_crop(crop, occlusion, noise, illum, rng)
            crops_pert.append(crop)
        # Padding hasta MAX_OBJ
        positions = np.zeros((MAX_OBJ, 3), dtype=np.float32)
        mask = np.zeros(MAX_OBJ, dtype=np.float32)
        for k, obj in enumerate(s["objects"]):
            positions[k] = obj["pos"]
            mask[k] = 1.0

        # CLIP encode
        text_emb = clip.encode_text([s["text"]])
        vis_emb_list = clip.encode_images(crops_pert)
        # Pad vis_emb to MAX_OBJ slots
        vis_padded = torch.zeros(1, MAX_OBJ, 768, device=device, dtype=torch.float32)
        vis_padded[0, :len(vis_emb_list)] = vis_emb_list
        mask_t = torch.tensor(mask, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            gates = gate(text_emb, vis_padded, mask_t)
        chosen = int(gates.argmax(dim=-1).item())
        ok = chosen == s["target_idx"]
        correct[i] = int(ok)
        confidences.append(float(gates.max().item()))

    return correct, np.array(confidences)


def main():
    print("[exp25] Robustez CLIP-image con domain randomization")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    clip, proj, gate = load_model(device)
    print(f"  device={device}")

    # Generar 300 escenas val (se reutilizan)
    rng_scenes = np.random.default_rng(99)
    print("  Generando 300 escenas val...")
    scenes = [generate_scene(rng_scenes) for _ in range(300)]

    # Condiciones a evaluar
    conditions = []
    # Oclusion (sin otros)
    for occ in [0.0, 0.2, 0.4, 0.6]:
        conditions.append({"name": f"occ_{int(occ*100)}",
                            "occlusion": occ, "noise": 0.0, "illum": 1.0,
                            "category": "occlusion", "level": occ})
    # Ruido
    for ns in [10, 25, 50]:
        conditions.append({"name": f"noise_{ns}",
                            "occlusion": 0.0, "noise": ns, "illum": 1.0,
                            "category": "noise", "level": ns})
    # Iluminacion
    for il in [0.5, 0.75, 1.5, 2.0]:
        conditions.append({"name": f"illum_{il}",
                            "occlusion": 0.0, "noise": 0.0, "illum": il,
                            "category": "illum", "level": il})
    # Combinacion (extrema realista)
    conditions.append({"name": "combined_realistic",
                        "occlusion": 0.2, "noise": 15, "illum": 0.8,
                        "category": "combined", "level": -1})

    results = {"conditions": [], "scene_n": len(scenes)}
    print(f"\n  Evaluando {len(conditions)} condiciones x {len(scenes)} escenas...")

    for cond in conditions:
        t0 = time.time()
        correct, conf = evaluate_condition(scenes, clip, gate,
                                              cond["occlusion"], cond["noise"], cond["illum"],
                                              device, seed=42 + hash(cond["name"]) % 1000)
        elapsed = time.time() - t0
        acc_mean = float(correct.mean())
        # Bootstrap CI 95% sobre acc usando bop-bootstrap-ci
        ci = bootstrap_recall(correct, threshold=0.5, B=1000, seed=42)
        # bootstrap_recall(<0.5) no es lo que queremos — necesitamos bootstrap sobre la media de correct
        # Usemos bootstrap_ci general con statistic=mean
        from bop_bootstrap_ci import bootstrap_ci
        ci_mean = bootstrap_ci(correct, statistic=np.mean, B=1000, seed=42)

        result = {
            **cond,
            "accuracy": acc_mean,
            "accuracy_ci95": ci_mean.as_dict(),
            "mean_confidence": float(np.mean(conf)),
            "n": len(scenes),
            "time_s": elapsed,
        }
        results["conditions"].append(result)
        ci_str = f"[{ci_mean.lo:.3f}, {ci_mean.hi:.3f}]"
        print(f"    {cond['name']:25s} acc={acc_mean:.3f} {ci_str}  conf={np.mean(conf):.2%}  {elapsed:.1f}s")

    # Resumen
    accs = [c["accuracy"] for c in results["conditions"]]
    results["summary"] = {
        "n_conditions": len(conditions),
        "mean_accuracy_across_conditions": float(np.mean(accs)),
        "min_accuracy": float(min(accs)),
        "max_accuracy": float(max(accs)),
        "robust_above_75pct": int(sum(1 for a in accs if a >= 0.75)),
    }

    with open(OUTPUT / "exp25_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] {OUTPUT / 'exp25_results.json'}")
    print(f"\n  Resumen: media={np.mean(accs):.1%} | min={min(accs):.1%} | max={max(accs):.1%}")
    print(f"  Robusto >= 75%: {results['summary']['robust_above_75pct']}/{len(conditions)} condiciones")

    # Plot curvas de robustez
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Occlusion
    occ_data = [c for c in results["conditions"] if c["category"] == "occlusion"]
    occ_data.sort(key=lambda c: c["level"])
    occ_x = [c["level"] * 100 for c in occ_data]
    occ_y = [c["accuracy"] for c in occ_data]
    occ_lo = [c["accuracy_ci95"]["lo"] for c in occ_data]
    occ_hi = [c["accuracy_ci95"]["hi"] for c in occ_data]
    axes[0].errorbar(occ_x, occ_y, yerr=[np.array(occ_y) - occ_lo, np.array(occ_hi) - occ_y],
                       fmt="o-", color="#FF6B35", linewidth=2, markersize=8, capsize=5)
    axes[0].set_xlabel("Oclusion (%)")
    axes[0].set_ylabel("Selection accuracy")
    axes[0].set_title("Robustez ante oclusion")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.75, color="red", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    # Noise
    noise_data = [c for c in results["conditions"] if c["category"] == "noise"]
    noise_data.sort(key=lambda c: c["level"])
    n_x = [0] + [c["level"] for c in noise_data]
    n_y = [results["conditions"][0]["accuracy"]] + [c["accuracy"] for c in noise_data]
    n_lo = [results["conditions"][0]["accuracy_ci95"]["lo"]] + [c["accuracy_ci95"]["lo"] for c in noise_data]
    n_hi = [results["conditions"][0]["accuracy_ci95"]["hi"]] + [c["accuracy_ci95"]["hi"] for c in noise_data]
    axes[1].errorbar(n_x, n_y, yerr=[np.array(n_y) - n_lo, np.array(n_hi) - n_y],
                       fmt="o-", color="#0098CD", linewidth=2, markersize=8, capsize=5)
    axes[1].set_xlabel("Ruido gaussiano sigma")
    axes[1].set_ylabel("Selection accuracy")
    axes[1].set_title("Robustez ante ruido sensor")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.75, color="red", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    # Illumination
    illum_data = [c for c in results["conditions"] if c["category"] == "illum"]
    illum_data.sort(key=lambda c: c["level"])
    levels = [c["level"] for c in illum_data]
    # Insertar el baseline 1.0
    levels_full = sorted(levels + [1.0])
    accs_full = []
    los, his = [], []
    for lv in levels_full:
        if lv == 1.0:
            accs_full.append(results["conditions"][0]["accuracy"])
            los.append(results["conditions"][0]["accuracy_ci95"]["lo"])
            his.append(results["conditions"][0]["accuracy_ci95"]["hi"])
        else:
            c = [c for c in illum_data if c["level"] == lv][0]
            accs_full.append(c["accuracy"])
            los.append(c["accuracy_ci95"]["lo"])
            his.append(c["accuracy_ci95"]["hi"])
    axes[2].errorbar(levels_full, accs_full,
                       yerr=[np.array(accs_full) - np.array(los), np.array(his) - np.array(accs_full)],
                       fmt="o-", color="#35876B", linewidth=2, markersize=8, capsize=5)
    axes[2].set_xlabel("Gain de iluminacion (1.0 = baseline)")
    axes[2].set_ylabel("Selection accuracy")
    axes[2].set_title("Robustez ante iluminacion")
    axes[2].set_ylim(0, 1.05)
    axes[2].axhline(0.75, color="red", linestyle="--", alpha=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("CLIP-image visual gate — Curvas de robustez (bootstrap CI 95%)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_png = OUTPUT / "fig_robustness_curves.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] {out_png}")


if __name__ == "__main__":
    main()
