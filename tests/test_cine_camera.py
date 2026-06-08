"""Tests para los helpers de geometría de src/simulation/cine_camera.py."""
import math

import numpy as np

from src.simulation.cine_camera import choreograph, lerp, look_at_matrix, orbit_position


def test_lerp_endpoints_and_mid():
    assert lerp(0.0, 10.0, 0.0) == 0.0
    assert lerp(0.0, 10.0, 1.0) == 10.0
    assert lerp(0.0, 10.0, 0.5) == 5.0


def test_orbit_position_radius_and_height():
    center = (0.0, 0.0, 0.0)
    p = orbit_position(center, radius=0.8, angle_rad=0.0, height=0.5)
    assert abs(p[0] - 0.8) < 1e-6
    assert abs(p[1] - 0.0) < 1e-6
    assert abs(p[2] - 0.5) < 1e-6
    p90 = orbit_position(center, radius=0.8, angle_rad=math.pi / 2, height=0.5)
    assert abs(math.hypot(p90[0], p90[1]) - 0.8) < 1e-6


def test_look_at_matrix_z_axis_points_at_target():
    # +Z (índices 2,6,10) debe ser el vector normalizado pos→target
    m = look_at_matrix((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    zc = (m[2], m[6], m[10])
    # cámara arriba mirando abajo → forward = (0,0,-1)
    assert abs(zc[0] - 0.0) < 1e-9
    assert abs(zc[1] - 0.0) < 1e-9
    assert abs(zc[2] - (-1.0)) < 1e-9


def test_look_at_matrix_orthonormal_and_position():
    m = look_at_matrix((0.6, -0.6, 0.7), (0.0, -0.30, 0.10))
    Xc = np.array([m[0], m[4], m[8]])
    Yc = np.array([m[1], m[5], m[9]])
    Zc = np.array([m[2], m[6], m[10]])
    # ortonormalidad
    for v in (Xc, Yc, Zc):
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6
    assert abs(np.dot(Xc, Yc)) < 1e-6
    assert abs(np.dot(Xc, Zc)) < 1e-6
    assert abs(np.dot(Yc, Zc)) < 1e-6
    # rotación propia (frame derecha-mano): det = +1, no espejado
    R = np.array([
        [m[0], m[1], m[2]],
        [m[4], m[5], m[6]],
        [m[8], m[9], m[10]],
    ])
    assert abs(np.linalg.det(R) - 1.0) < 1e-6
    # última columna = posición
    assert (m[3], m[7], m[11]) == (0.6, -0.6, 0.7)


def test_choreograph_clamps_progress_out_of_range():
    tcp = (0.45, -0.12, 0.20)
    center = (0.3, -0.3, 0.1)
    # progress fuera de [0,1] se clampea: <0 == 0.0, >1 == 1.0
    assert choreograph(-0.5, tcp, center) == choreograph(0.0, tcp, center)
    assert choreograph(1.5, tcp, center) == choreograph(1.0, tcp, center)


def test_choreograph_phases_move_camera_closer():
    tcp = (0.45, -0.12, 0.20)
    center = (0.3, -0.3, 0.1)
    cam_far, _ = choreograph(0.0, tcp, center)
    cam_near, _ = choreograph(0.5, tcp, center)
    cam_pull, _ = choreograph(1.0, tcp, center)
    d_near_tcp = math.dist(cam_near, tcp)
    # en seguimiento la cámara está más cerca del TCP que en establecimiento
    assert d_near_tcp < math.dist(cam_far, tcp)
    # en retroceso vuelve a alejarse del TCP respecto al seguimiento
    assert math.dist(cam_pull, tcp) > d_near_tcp
