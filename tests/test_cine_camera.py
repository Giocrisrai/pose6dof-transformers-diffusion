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


def test_look_at_euler_points_down_toward_target():
    eul = look_at_euler(cam_pos=(0.0, 0.0, 1.0), target=(0.0, 0.0, 0.0))
    assert len(eul) == 3
    assert all(isinstance(v, float) for v in eul)


def test_choreograph_phases_move_camera_closer():
    tcp = (0.45, -0.12, 0.20)
    center = (0.3, -0.3, 0.1)
    cam_far, tgt0 = choreograph(0.0, tcp, center)
    cam_near, tgt1 = choreograph(0.5, tcp, center)
    cam_pull, tgt2 = choreograph(1.0, tcp, center)
    d_near = math.dist(cam_near, tcp)
    d_pull = math.dist(cam_pull, center)
    assert d_near < math.dist(cam_far, tcp)
    assert d_pull > d_near
