"""Tests para los helpers de geometría de src/simulation/cine_camera.py."""
import math

from src.simulation.cine_camera import choreograph, lerp, look_at_euler, orbit_position


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


def test_look_at_euler_camara_arriba_mira_abajo():
    # cam en (0,0,1) mirando al origen → d=(0,0,-1): yaw=0, pitch=-pi/2
    # → alpha=0, beta=0, gamma=pi/2
    eul = look_at_euler(cam_pos=(0.0, 0.0, 1.0), target=(0.0, 0.0, 0.0))
    assert abs(eul[0] - 0.0) < 1e-9
    assert abs(eul[1] - 0.0) < 1e-9
    assert abs(eul[2] - math.pi / 2) < 1e-9


def test_look_at_euler_horizontal():
    # cam en +y mirando al origen → d=(0,-1,0): yaw=-pi/2, pitch=0
    # → alpha=-pi/2, beta=0, gamma=0
    eul = look_at_euler(cam_pos=(0.0, 1.0, 0.0), target=(0.0, 0.0, 0.0))
    assert abs(eul[0] - (-math.pi / 2)) < 1e-9
    assert abs(eul[1] - 0.0) < 1e-9
    assert abs(eul[2] - 0.0) < 1e-9


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
