"""Tests unitarios para src/control/pbvs.py."""
import numpy as np
import pytest
from src.control import PBVSController, simulate_pbvs_loop, se3_error, so3_log


def make_T(R=None, t=None):
    """Construye matriz SE(3) 4x4."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class TestSO3Log:
    def test_identity_returns_zero(self):
        assert np.allclose(so3_log(np.eye(3)), np.zeros(3))

    def test_z_rotation(self):
        angle = np.pi / 4
        omega = so3_log(rotation_z(angle))
        assert np.allclose(omega, [0, 0, angle], atol=1e-9)

    def test_x_rotation(self):
        angle = 0.5
        R = np.array([[1, 0, 0],
                       [0, np.cos(angle), -np.sin(angle)],
                       [0, np.sin(angle),  np.cos(angle)]])
        omega = so3_log(R)
        assert np.allclose(omega, [angle, 0, 0], atol=1e-9)

    def test_pi_rotation(self):
        # Rotación de pi en torno a z
        R = rotation_z(np.pi)
        omega = so3_log(R)
        assert np.allclose(np.linalg.norm(omega), np.pi, atol=1e-6)


class TestSE3Error:
    def test_identical_poses_zero_error(self):
        T = make_T(rotation_z(0.5), [0.1, 0.2, 0.3])
        xi = se3_error(T, T)
        assert np.allclose(xi, np.zeros(6), atol=1e-9)

    def test_pure_translation_error(self):
        T1 = make_T(t=[0, 0, 0])
        T2 = make_T(t=[0.1, 0.0, 0.0])
        xi = se3_error(T1, T2)
        assert np.allclose(xi[:3], [0.1, 0.0, 0.0], atol=1e-9)
        assert np.allclose(xi[3:], np.zeros(3), atol=1e-9)

    def test_pure_rotation_error(self):
        T1 = np.eye(4)
        T2 = make_T(R=rotation_z(0.3))
        xi = se3_error(T1, T2)
        assert np.allclose(xi[:3], np.zeros(3), atol=1e-9)
        assert np.allclose(xi[3:], [0, 0, 0.3], atol=1e-9)


class TestPBVSController:
    def test_initialization_defaults(self):
        c = PBVSController()
        assert c.kp_lin > 0
        assert c.kp_ang > 0
        assert c.v_max > 0
        assert c.eps_lin > 0

    def test_zero_error_converged(self):
        c = PBVSController()
        T = np.eye(4)
        xi_cmd, info = c.step(T, T)
        assert info["converged"]
        assert info["error_lin"] < 1e-9
        assert info["error_ang"] < 1e-9
        assert np.allclose(xi_cmd, np.zeros(6), atol=1e-9)

    def test_proportional_velocity(self):
        c = PBVSController(kp_lin=2.0, kp_ang=2.0, v_max=10.0, w_max=10.0)
        T_target = make_T(t=[0.05, 0, 0])
        xi_cmd, info = c.step(np.eye(4), T_target)
        # Velocidad lineal x debe ser kp_lin * 0.05 = 0.1 m/s
        assert np.isclose(xi_cmd[0], 0.1, atol=1e-9)
        assert not info["converged"]

    def test_velocity_saturation(self):
        c = PBVSController(kp_lin=10.0, v_max=0.1)
        T_target = make_T(t=[1.0, 0, 0])  # error muy grande
        xi_cmd, info = c.step(np.eye(4), T_target)
        v_norm = np.linalg.norm(xi_cmd[:3])
        assert v_norm <= c.v_max + 1e-9


class TestSimulatePBVSLoop:
    def test_converges_pure_translation(self):
        T_init = np.eye(4)
        T_target = make_T(t=[0.1, 0.05, 0.02])
        result = simulate_pbvs_loop(T_init, T_target, dt=0.05, max_iters=200)
        assert result["converged"]
        assert result["n_iters"] < 200

    def test_converges_pure_rotation(self):
        T_init = np.eye(4)
        T_target = make_T(R=rotation_z(0.5))
        result = simulate_pbvs_loop(T_init, T_target, dt=0.05, max_iters=200)
        assert result["converged"]

    def test_converges_combined(self):
        T_init = np.eye(4)
        T_target = make_T(R=rotation_z(0.3), t=[0.05, -0.03, 0.02])
        result = simulate_pbvs_loop(T_init, T_target, dt=0.05, max_iters=200)
        assert result["converged"]

    def test_error_decreases_monotonically(self):
        T_init = np.eye(4)
        T_target = make_T(t=[0.1, 0, 0])
        result = simulate_pbvs_loop(T_init, T_target, dt=0.05, max_iters=200)
        errors = result["errors_lin_m"]
        # Error debe decrecer al menos en general (puede haber pequeñas oscilaciones)
        assert errors[-1] < errors[0]
        # Decreciente en al menos 80% de los pasos consecutivos
        decreasing = sum(1 for i in range(1, len(errors)) if errors[i] <= errors[i-1])
        assert decreasing / max(len(errors)-1, 1) >= 0.8

    def test_does_not_converge_unreachable(self):
        # Tolerancia imposible
        from src.control import PBVSController
        c = PBVSController(eps_lin=1e-15, eps_ang=1e-15)
        T_init = np.eye(4)
        T_target = make_T(t=[0.1, 0, 0])
        result = simulate_pbvs_loop(T_init, T_target, dt=0.05, max_iters=20, controller=c)
        assert not result["converged"]
        assert result["n_iters"] == 20
