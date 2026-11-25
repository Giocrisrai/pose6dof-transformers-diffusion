"""Bandeja de depósito para la simulación.

En la escena base la pieza se suelta sobre una superficie abierta: una esfera o
un cilindro ruedan fuera de la zona de depósito. Una celda real deposita en un
contenedor; esta bandeja (suelo + cuatro paredes, estática) lo reproduce.
"""
from __future__ import annotations

LADO_INTERIOR_M = 0.30
ALTO_PARED_M = 0.16
GROSOR_M = 0.01


def piezas_bandeja(
    centro_xy: tuple[float, float],
    z_base: float = 0.0,
    lado: float = LADO_INTERIOR_M,
    alto: float = ALTO_PARED_M,
    grosor: float = GROSOR_M,
) -> list[tuple[list[float], list[float]]]:
    """Geometría de la bandeja como lista de (tamaño [x, y, z], centro [x, y, z]).

    El hueco interior mide `lado` x `lado` y queda centrado en `centro_xy`; el
    suelo apoya en `z_base` y las paredes suben `alto` sobre el suelo.
    """
    cx, cy = centro_xy
    medio = lado / 2 + grosor / 2
    exterior = lado + 2 * grosor
    z_pared = z_base + grosor + alto / 2
    return [
        ([exterior, exterior, grosor], [cx, cy, z_base + grosor / 2]),      # suelo
        ([grosor, exterior, alto], [cx - medio, cy, z_pared]),
        ([grosor, exterior, alto], [cx + medio, cy, z_pared]),
        ([exterior, grosor, alto], [cx, cy - medio, z_pared]),
        ([exterior, grosor, alto], [cx, cy + medio, z_pared]),
    ]


def crear_bandeja_deposito(
    sim,
    centro_xy: tuple[float, float],
    z_base: float = 0.0,
    visible_para_camaras: bool = False,
) -> list[int]:
    """Crea la bandeja en la escena cargada y devuelve los handles de sus piezas.

    Por defecto no se renderiza en las cámaras: las políticas visuales se
    entrenaron sin ella y cualquier cambio en la imagen degrada su puntería
    (ver exploración 15). La física de la bandeja no cambia.
    """
    handles = []
    for tam, centro in piezas_bandeja(centro_xy, z_base):
        h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, tam, 0)
        sim.setObjectAlias(h, "bandeja_deposito")
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [0.25, 0.25, 0.3])
        if not visible_para_camaras:
            sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
        sim.setObjectPosition(h, -1, centro)
        handles.append(h)
    return handles
