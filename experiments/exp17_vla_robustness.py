#!/usr/bin/env python3
"""Experimento 17: robustez del VLA-lite ante variaciones linguisticas.

Verifica que el modelo CLIP+gate del exp16 generaliza a frases NUNCA vistas
durante el training (que usaba solo 5 templates simples). Mide degradacion
con frases mas complejas, sinonimos, modificadores, etc.

Esto es importante para saber si el modelo realmente entiende el lenguaje o
solo aprendio plantilla "el {color} aparece en la frase".

Salida:
    experiments/results/exp17_vla_robustness/exp17_results.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import ConditionalUNet1D


OUTPUT = REPO / "experiments/results/exp17_vla_robustness"
OUTPUT.mkdir(parents=True, exist_ok=True)


# Familias de frases de test, agrupadas por dificultad creciente
TEST_FAMILIES = {
    "1_in_distribution": {
        "description": "Templates similares a los del training (5 originales)",
        "templates": [
            "pick the {color} object",
            "grab the {color} item",
            "get the {color} thing",
            "select the {color} one",
            "take the {color} cube",
        ],
    },
    "2_synonyms": {
        "description": "Sinonimos de verbos no vistos en training",
        "templates": [
            "fetch the {color} object",
            "retrieve the {color} item",
            "lift the {color} cube",
            "collect the {color} one",
            "obtain the {color} thing",
        ],
    },
    "3_modifiers": {
        "description": "Frases con modificadores y articulos extra",
        "templates": [
            "please pick the {color} object",
            "I want the {color} one",
            "go for the {color} item",
            "the {color} cube is what I need",
            "give me the {color} thing",
        ],
    },
    "4_longer_sentences": {
        "description": "Frases mas largas con contexto",
        "templates": [
            "between the two, pick the {color} object",
            "ignore the others and grab the {color} item",
            "your target is the {color} one",
            "go ahead and take the {color} cube please",
            "the goal here is to fetch the {color} thing",
        ],
    },
    "5_color_with_object": {
        "description": "Color seguido de nombre concreto (cup, ball, box)",
        "templates": [
            "pick the {color} cup",
            "grab the {color} ball",
            "select the {color} box",
            "take the {color} sphere",
            "get the {color} item",
        ],
    },
    "6_implicit": {
        "description": "Casos limite: el color es implicito por contraste",
        "templates": [
            "not the other one, the {color}",
            "pick anything {color}",
            "the {color}, please",
            "{color}",
            "go {color}",
        ],
    },
}


def load_vla_model():
    """Carga el modelo VLA-lite del exp16."""
    from transformers import CLIPTokenizer, CLIPTextModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_mod = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    ckpt = torch.load(REPO / "data/models/diffusion_policy_clip.pth",
                       map_location=device, weights_only=True)

    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.score = nn.Sequential(
                nn.Linear(515, 128), nn.Mish(),
                nn.Linear(128, 128), nn.Mish(),
                nn.Linear(128, 1),
            )

        def forward(self, c, a, b):
            sa = self.score(torch.cat([a, c], -1)).squeeze(-1)
            sb = self.score(torch.cat([b, c], -1)).squeeze(-1)
            return F.softmax(torch.stack([sa, sb], -1), dim=-1).unbind(-1)

    gate = Gate().to(device).eval()
    gate.load_state_dict(ckpt["gate_state_dict"])
    return tok, clip_mod, gate, device


def encode_text(tok, clip_mod, texts, device):
    with torch.no_grad():
        ins = tok(texts, padding=True, return_tensors="pt", truncation=True)
        ins = {k: v.to(device) for k, v in ins.items()}
        return clip_mod(**ins).pooler_output


def gate_decision(gate, clip_emb, color_a, color_b, device):
    RGB = {"red": [1.0, 0.0, 0.0], "blue": [0.0, 0.0, 1.0], "green": [0.0, 1.0, 0.0]}
    ra = torch.tensor([RGB[color_a]], device=device, dtype=torch.float32)
    rb = torch.tensor([RGB[color_b]], device=device, dtype=torch.float32)
    with torch.no_grad():
        ga, gb = gate(clip_emb, ra, rb)
    return float(ga.item()), float(gb.item())


def evaluate_family(family, tok, clip_mod, gate, device, n_per_template=30, seed=42):
    """Para cada template, genera n_per_template escenas aleatorias y mide acc."""
    rng = np.random.default_rng(seed)
    colors = ["red", "blue", "green"]
    correct = 0
    total = 0
    confidences = []
    sample_fails = []

    for template in family["templates"]:
        for _ in range(n_per_template):
            # Sample 2 colores distintos
            ca, cb = rng.choice(colors, size=2, replace=False)
            ca, cb = str(ca), str(cb)
            # Target color = uno al azar de ca/cb
            target_idx = rng.integers(0, 2)
            target_color = ca if target_idx == 0 else cb
            text = template.format(color=target_color)
            ce = encode_text(tok, clip_mod, [text], device)
            ga, gb = gate_decision(gate, ce, ca, cb, device)
            chosen = 0 if ga > gb else 1
            confidence = max(ga, gb)
            confidences.append(confidence)
            if chosen == target_idx:
                correct += 1
            else:
                if len(sample_fails) < 3:
                    sample_fails.append({
                        "text": text, "color_a": ca, "color_b": cb,
                        "target": target_color, "chosen_color": ca if chosen == 0 else cb,
                        "gate_a": ga, "gate_b": gb,
                    })
            total += 1

    return {
        "accuracy": correct / total,
        "n": total,
        "mean_confidence": float(np.mean(confidences)),
        "min_confidence": float(np.min(confidences)),
        "sample_fails": sample_fails,
    }


def main():
    print("[exp17] Evaluando robustez VLA-lite ante variaciones linguisticas")
    tok, clip_mod, gate, device = load_vla_model()
    print(f"  device={device}")

    results = {"families": {}}

    for fam_name, family in TEST_FAMILIES.items():
        print(f"\n  -> {fam_name}: {family['description']}")
        t0 = time.time()
        r = evaluate_family(family, tok, clip_mod, gate, device, n_per_template=30)
        elapsed = time.time() - t0
        r["description"] = family["description"]
        r["templates"] = family["templates"]
        r["eval_time_s"] = elapsed
        results["families"][fam_name] = r
        print(f"     accuracy={r['accuracy']:.1%} (n={r['n']}) | "
              f"mean conf={r['mean_confidence']:.2%} | min conf={r['min_confidence']:.2%} | "
              f"{elapsed:.1f}s")
        for fail in r["sample_fails"][:2]:
            print(f"     fail: '{fail['text']}' (target={fail['target']}, chose={fail['chosen_color']})")

    # Resumen
    accuracies = [r["accuracy"] for r in results["families"].values()]
    results["summary"] = {
        "mean_accuracy": float(np.mean(accuracies)),
        "min_accuracy": float(np.min(accuracies)),
        "max_accuracy": float(np.max(accuracies)),
        "robust_above_75pct": int(sum(1 for a in accuracies if a >= 0.75)),
        "total_families": len(accuracies),
    }

    out_json = OUTPUT / "exp17_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  RESUMEN: media {results['summary']['mean_accuracy']:.1%} | "
          f"min {results['summary']['min_accuracy']:.1%} | "
          f"max {results['summary']['max_accuracy']:.1%}")
    print(f"  Familias >= 75 % acc: {results['summary']['robust_above_75pct']}/{results['summary']['total_families']}")
    print(f"  [OK] {out_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
