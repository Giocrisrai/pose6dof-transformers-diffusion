"""Utilidades compartidas del subsistema de simulación."""
from __future__ import annotations


def map_fp_pose_to_sim_workspace(t_pred):
    """Mapea el centroide de una pose FP (YCB-V dataset frame) al
    workspace del UR5 en bin_base.ttt.

    Las poses FP están en el frame de la cámara del dataset (típicamente
    t_pred[2] ≈ 0.5-1.5 m). Para el demo del sim, usamos solo la
    componente XY de la pose para variar dentro del workspace (Z = cube
    height fija).

    Args:
        t_pred: lista o array de 3 floats (translation de la pose FP).

    Returns:
        list[float] de 3 elementos: [x, y, z] en coords mundo del sim.
    """
    x_offset = max(-0.05, min(0.05, t_pred[0]))
    y_offset = max(-0.05, min(0.05, t_pred[1]))
    return [0.46 + x_offset, -0.10 + y_offset, 0.033]
