"""Léxico controlado ES/EN y normalizadores.

Mapea sinónimos en español e inglés a un valor canónico en inglés
(coherente con los atributos usados en exp16-26).
"""
from __future__ import annotations

from typing import Optional

COLORS = {
    "red": ["red", "rojo", "roja"],
    "blue": ["blue", "azul"],
    "green": ["green", "verde"],
    "yellow": ["yellow", "amarillo", "amarilla"],
}
SHAPES = {
    "cube": ["cube", "cubo", "caja cuadrada"],
    "sphere": ["sphere", "esfera", "bola", "ball"],
    "cylinder": ["cylinder", "cilindro"],
    "box": ["box", "caja"],
}
SIZES = {
    "small": ["small", "pequeño", "pequeña", "chico", "little"],
    "large": ["large", "big", "grande", "gran"],
}
RELATIONS = {
    "left_of": ["left", "izquierda", "a la izquierda", "left of"],
    "right_of": ["right", "derecha", "a la derecha", "right of"],
    "nearest": ["nearest", "closest", "más cercano", "mas cercano", "cercano"],
    "farthest": ["farthest", "furthest", "más lejano", "mas lejano", "lejano"],
    "on_top": ["on top", "encima", "arriba", "top"],
}
NOUNS = ["object", "objeto", "piece", "pieza", "block", "bloque", "item"]


def _match(text: str, table: dict) -> Optional[str]:
    t = text.lower()
    # buscar la coincidencia más larga primero (frases > palabras)
    best = None
    best_len = 0
    for canonical, synonyms in table.items():
        for syn in synonyms:
            if syn in t and len(syn) > best_len:
                best, best_len = canonical, len(syn)
    return best


def normalize_color(text: str) -> Optional[str]:
    return _match(text, COLORS)


def normalize_shape(text: str) -> Optional[str]:
    return _match(text, SHAPES)


def normalize_size(text: str) -> Optional[str]:
    return _match(text, SIZES)


def normalize_relation(text: str) -> Optional[str]:
    return _match(text, RELATIONS)
