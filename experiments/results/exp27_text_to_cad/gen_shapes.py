#!/usr/bin/env python3
"""exp27 · text-to-CAD — genera un catálogo de piezas de test paramétricas.

Cada pieza traduce una descripción a un sólido con geometría de ground-truth
exacta, exportado a STL/OBJ para simulación y pose.
"""
from pathlib import Path

import trimesh
from build123d import Box, Circle, Cylinder, Pos, RegularPolygon, Rot, export_stl, extrude

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)


def escuadra_L():
    """Escuadra en L 60x40x45 mm con agujeros (asimétrica)."""
    L, W, T, H, r = 60.0, 40.0, 5.0, 45.0, 3.0
    p = Pos(0, 0, T/2) * Box(L, W, T)
    p += Pos(-L/2 + T/2, 0, H/2) * Box(T, W, H)
    for hx, hy in [(15, 12), (15, -12)]:
        p -= Pos(hx, hy, T/2) * Cylinder(radius=r, height=T*2)
    p -= Pos(-L/2 + T/2, 0, H-12) * (Rot(0, 90, 0) * Cylinder(radius=r, height=T*2))
    return p


def tuerca_hex():
    """Tuerca hexagonal grande: 44 mm entre vértices, 18 mm alto, agujero 20 mm (simetría 6)."""
    hexp = extrude(RegularPolygon(22, 6), 18)
    hexp -= extrude(Circle(10), 18)
    return Pos(0, 0, 9) * hexp   # apoyada en z=0


def bloque_escalonado():
    """Bloque en dos escalones desalineados (asimétrico)."""
    b = Pos(0, 0, 8) * Box(70, 45, 16)         # base z[0,16]
    b += Pos(-16, 0, 24) * Box(38, 45, 16)     # escalón z[16,32], offset -x
    return b


SHAPES = {
    "test_bracket": escuadra_L,
    "hex_nut": tuerca_hex,
    "stepped_block": bloque_escalonado,
}

if __name__ == "__main__":
    for name, fn in SHAPES.items():
        part = fn()
        stl = ASSETS / f"{name}.stl"
        export_stl(part, str(stl))
        m = trimesh.load(stl)
        m.export(ASSETS / f"{name}.obj")
        bb = m.extents * 1000
        print(f"{name:16s} vol={part.volume:8.1f}mm^3  bbox={bb.round(0).tolist()}mm  "
              f"watertight={m.is_watertight}  tris={len(m.faces)}")
