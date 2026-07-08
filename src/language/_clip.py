"""Clasificación zero-shot de atributos con CLIP (opcional, lazy).

Recorta el bbox de un objeto y compara contra prompts de color/forma/tamaño
con CLIP image-text (exp24). Sólo se importa si method="clip_image".
"""
from __future__ import annotations

from functools import lru_cache

from src.language.vocab import COLORS, SHAPES, SIZES


def _crop(rgb, bbox):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    return rgb[y1:y2, x1:x2]


@lru_cache(maxsize=2)
def _load_clip(model_name: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    return model, proc, torch


def _best_label(crop, table, model_name):
    model, proc, torch = _load_clip(model_name)
    labels = list(table.keys())
    prompts = [f"a photo of a {label} object" for label in labels]
    inputs = proc(text=prompts, images=crop, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image.softmax(dim=1)[0]
    return labels[int(logits.argmax())]


def clip_attributes(rgb, bbox, model_name: str = "openai/clip-vit-base-patch32") -> dict:
    """Devuelve {"color","shape","size"} estimados por CLIP para un objeto."""
    crop = _crop(rgb, bbox)
    return {
        "color": _best_label(crop, COLORS, model_name),
        "shape": _best_label(crop, SHAPES, model_name),
        "size": _best_label(crop, SIZES, model_name),
    }
