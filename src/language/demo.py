"""Escena sintética fija para demos de lenguaje natural (CLI y dashboard).

Tres objetos coherentes con exp16/24: cubo rojo (izquierda), cubo azul
(centro), esfera roja (derecha). Centraliza la escena para que la CLI y el
dashboard no diverjan.
"""
from __future__ import annotations

from src.language.schema import ObjectView


def demo_scene() -> list[ObjectView]:
    """Devuelve la escena demo de 3 objetos."""
    return [
        ObjectView(0, (-0.20, 0.0, 0.5), {"color": "red", "shape": "cube", "size": "large"}),
        ObjectView(1, (0.00, 0.0, 0.5), {"color": "blue", "shape": "cube", "size": "small"}),
        ObjectView(2, (0.20, 0.0, 0.5), {"color": "red", "shape": "sphere", "size": "small"}),
    ]
