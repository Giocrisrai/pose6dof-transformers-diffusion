"""Cámara cinematográfica dedicada para el showcase reel.

Vision sensor de alta resolución, SEPARADO de la cámara de percepción
(/rgb_camera), creado en runtime. La de percepción queda fija para no
corromper el RGB-D que recibe la Diffusion Policy.

Dos partes:
- Helpers de geometría puros (lerp, orbit_position, look_at_euler, choreograph):
  testeables sin simulador.
- Clase CineCamera: ciclo de vida del vision sensor + captura (tarea posterior).
"""
from __future__ import annotations

import math

import numpy as np

Vec3 = tuple[float, float, float]


def lerp(a: float, b: float, t: float) -> float:
    """Interpolación lineal a→b para t en [0,1]."""
    return a + (b - a) * t


def orbit_position(center: Vec3, radius: float, angle_rad: float, height: float) -> Vec3:
    """Posición sobre un círculo de radio `radius` alrededor de `center` en el
    plano xy, a altura absoluta `height`. angle_rad=0 → dirección +x."""
    return (
        center[0] + radius * math.cos(angle_rad),
        center[1] + radius * math.sin(angle_rad),
        height,
    )


def look_at_euler(cam_pos: Vec3, target: Vec3) -> Vec3:
    """Ángulos de Euler (alpha,beta,gamma, convención XYZ de CoppeliaSim) para
    que una cámara en `cam_pos` mire hacia `target`. El eje -Z de la cámara
    apunta al target."""
    d = np.array(target, dtype=float) - np.array(cam_pos, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    d /= n
    yaw = math.atan2(d[1], d[0])
    pitch = math.asin(max(-1.0, min(1.0, d[2])))
    alpha = -(math.pi / 2 + pitch)
    beta = 0.0
    gamma = yaw + math.pi / 2
    return (float(alpha), float(beta), float(gamma))


def choreograph(progress: float, tcp: Vec3, workspace_center: Vec3) -> tuple[Vec3, Vec3]:
    """Devuelve (posición_cámara, target_lookat) para un `progress` en [0,1]
    del pick. Tres fases:
      - [0.00,0.35) establecimiento: órbita amplia mirando al workspace.
      - [0.35,0.70) seguimiento: acercamiento mirando al TCP.
      - [0.70,1.00] retroceso: alejamiento mirando al workspace.
    """
    p = max(0.0, min(1.0, progress))
    if p < 0.35:
        a = p / 0.35
        angle = lerp(math.radians(20), math.radians(80), a)
        pos = orbit_position(workspace_center, radius=lerp(1.0, 0.7, a),
                             angle_rad=angle, height=lerp(0.9, 0.7, a))
        return pos, workspace_center
    if p < 0.70:
        a = (p - 0.35) / 0.35
        angle = lerp(math.radians(80), math.radians(110), a)
        pos = orbit_position(tcp, radius=lerp(0.55, 0.32, a),
                             angle_rad=angle, height=lerp(0.55, 0.35, a))
        return pos, tcp
    a = (p - 0.70) / 0.30
    angle = lerp(math.radians(110), math.radians(150), a)
    pos = orbit_position(workspace_center, radius=lerp(0.5, 1.0, a),
                         angle_rad=angle, height=lerp(0.45, 0.85, a))
    return pos, workspace_center
