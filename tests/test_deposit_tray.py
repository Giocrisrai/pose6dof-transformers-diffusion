"""Geometría de la bandeja de depósito (sin simulador)."""
from types import SimpleNamespace

import numpy as np

from src.simulation.deposit_tray import (
    ALTO_PARED_M,
    GROSOR_M,
    LADO_INTERIOR_M,
    crear_bandeja_deposito,
    piezas_bandeja,
)


def test_bandeja_tiene_suelo_y_cuatro_paredes():
    piezas = piezas_bandeja((-0.3, -0.3))
    assert len(piezas) == 5
    suelo_tam, suelo_centro = piezas[0]
    assert suelo_tam[2] == GROSOR_M
    assert np.allclose(suelo_centro[:2], (-0.3, -0.3))


def test_hueco_interior_centrado_y_del_tamano_pedido():
    cx, cy = 0.1, -0.2
    paredes = piezas_bandeja((cx, cy))[1:]
    caras_x = sorted(c[0] - np.sign(c[0] - cx) * t[0] / 2 for t, c in paredes[:2])
    caras_y = sorted(c[1] - np.sign(c[1] - cy) * t[1] / 2 for t, c in paredes[2:])
    assert np.allclose(caras_x, (cx - LADO_INTERIOR_M / 2, cx + LADO_INTERIOR_M / 2))
    assert np.allclose(caras_y, (cy - LADO_INTERIOR_M / 2, cy + LADO_INTERIOR_M / 2))


def test_paredes_apoyan_en_el_suelo_y_suben_el_alto_pedido():
    z_base = 0.05
    for tam, centro in piezas_bandeja((0.0, 0.0), z_base=z_base)[1:]:
        assert np.isclose(centro[2] - tam[2] / 2, z_base + GROSOR_M)
        assert np.isclose(centro[2] + tam[2] / 2, z_base + GROSOR_M + ALTO_PARED_M)


def test_crear_bandeja_deja_piezas_estaticas_y_con_colision():
    llamadas = []
    contador = iter(range(100, 200))
    sim = SimpleNamespace(
        primitiveshape_cuboid=0, shapeintparam_static=1, shapeintparam_respondable=2,
        colorcomponent_ambient_diffuse=3, objintparam_visibility_layer=4,
        createPrimitiveShape=lambda tipo, tam, opc: next(contador),
        setObjectAlias=lambda h, a: None,
        setObjectInt32Param=lambda h, p, v: llamadas.append((h, p, v)),
        setShapeColor=lambda *a: None,
        setObjectPosition=lambda h, rel, pos: None,
    )
    handles = crear_bandeja_deposito(sim, (-0.3, -0.3))
    assert len(handles) == 5
    for h in handles:
        assert (h, 1, 1) in llamadas  # estática
        assert (h, 2, 1) in llamadas  # con colisión
        assert (h, 4, 0) in llamadas  # fuera de la imagen de las cámaras
