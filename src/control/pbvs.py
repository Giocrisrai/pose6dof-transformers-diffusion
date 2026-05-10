"""Position-Based Visual Servoing (PBVS) controller para el pipeline TFM.

Implementa control proporcional en el espacio cartesiano SE(3) que cierra
el lazo entre la pose estimada por FoundationPose y la pose objetivo
(obtenida del muestreo de Diffusion Policy).

El controlador computa una velocidad lineal y angular (xi en se(3) ≅ R^6)
proporcional al error en SE(3) calculado mediante la operacion logaritmica
del grupo de Lie.

Referencias:
- Chaumette & Hutchinson (2006). Visual Servo Control I: Basic Approaches. RA Magazine.
- Sola et al. (2018). A Micro Lie Theory for State Estimation in Robotics.
- Li et al. (2025). Enhancing PBVS through Transformer-Based RL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def so3_log(R: np.ndarray) -> np.ndarray:
    """Operador logaritmico SO(3) -> so(3) ~= R^3.

    Devuelve el vector axis*angle correspondiente a la rotacion R.
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if abs(theta) < 1e-9:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-9:
        # Caso angulo pi: usar diagonal
        diag = np.diag(R)
        i = int(np.argmax(diag))
        v = R[:, i] + np.eye(3)[:, i]
        v = v / np.linalg.norm(v)
        return theta * v
    return theta / (2.0 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


def se3_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """Error en SE(3): xi = log(T_current^-1 @ T_target) en se(3) ≅ R^6.

    Args:
        T_current: matriz 4x4 SE(3) actual
        T_target:  matriz 4x4 SE(3) objetivo

    Returns:
        xi: vector 6D [v, omega] donde v=traslacion, omega=axis*angle
    """
    T_err = np.linalg.inv(T_current) @ T_target
    R_err = T_err[:3, :3]
    t_err = T_err[:3, 3]
    omega = so3_log(R_err)
    return np.concatenate([t_err, omega])


@dataclass
class PBVSController:
    """Controlador PBVS proporcional en SE(3).

    Atributos:
        kp_lin: ganancia proporcional traslacion (1/s)
        kp_ang: ganancia proporcional rotacion (1/s)
        v_max:  saturacion de velocidad lineal (m/s)
        w_max:  saturacion de velocidad angular (rad/s)
        eps_lin: tolerancia traslacion para considerar convergido (m)
        eps_ang: tolerancia rotacion para convergencia (rad)
    """
    kp_lin: float = 1.5
    kp_ang: float = 1.5
    v_max: float = 0.25       # 25 cm/s
    w_max: float = 1.5        # ~85 deg/s
    eps_lin: float = 0.002    # 2 mm
    eps_ang: float = 0.01     # ~0.6 deg

    def step(self, T_current: np.ndarray, T_target: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Calcula la velocidad de comando para acercar T_current a T_target.

        Returns:
            xi_cmd: velocidad 6D [vx, vy, vz, wx, wy, wz]
            info: dict con error_lin, error_ang, converged
        """
        xi_err = se3_error(T_current, T_target)
        v_err, w_err = xi_err[:3], xi_err[3:]
        error_lin = float(np.linalg.norm(v_err))
        error_ang = float(np.linalg.norm(w_err))

        # Control proporcional
        v_cmd = self.kp_lin * v_err
        w_cmd = self.kp_ang * w_err

        # Saturacion
        v_norm = np.linalg.norm(v_cmd)
        if v_norm > self.v_max:
            v_cmd = v_cmd * (self.v_max / v_norm)
        w_norm = np.linalg.norm(w_cmd)
        if w_norm > self.w_max:
            w_cmd = w_cmd * (self.w_max / w_norm)

        xi_cmd = np.concatenate([v_cmd, w_cmd])
        converged = error_lin < self.eps_lin and error_ang < self.eps_ang

        info = {
            "error_lin": error_lin,
            "error_ang": error_ang,
            "v_cmd_norm": float(np.linalg.norm(v_cmd)),
            "w_cmd_norm": float(np.linalg.norm(w_cmd)),
            "converged": converged,
        }
        return xi_cmd, info


def pbvs_step(T_current: np.ndarray, T_target: np.ndarray,
              kp_lin: float = 1.5, kp_ang: float = 1.5) -> Tuple[np.ndarray, dict]:
    """API funcional: un paso de PBVS sin instanciar controlador."""
    ctrl = PBVSController(kp_lin=kp_lin, kp_ang=kp_ang)
    return ctrl.step(T_current, T_target)


def simulate_pbvs_loop(T_initial: np.ndarray, T_target: np.ndarray,
                       dt: float = 0.05, max_iters: int = 500,
                       controller: Optional[PBVSController] = None) -> dict:
    """Simula el bucle PBVS hasta convergencia o max_iters.

    Integracion del paso de SE(3) por interpolacion en se(3):
        T_{k+1} = T_k @ exp(dt * xi_cmd_k)

    Returns:
        dict con trayectoria, iteraciones, errores, convergencia
    """
    if controller is None:
        controller = PBVSController()

    T = T_initial.copy()
    trajectory = [T.copy()]
    errors_lin = []
    errors_ang = []

    converged_at = None
    for k in range(max_iters):
        xi_cmd, info = controller.step(T, T_target)
        errors_lin.append(info["error_lin"])
        errors_ang.append(info["error_ang"])

        if info["converged"]:
            converged_at = k
            break

        # Integrar: T_{k+1} = T_k @ exp(dt * xi_cmd)
        # Aproximacion lineal para xi pequeno (suficiente con dt=0.05)
        v, w = xi_cmd[:3], xi_cmd[3:]
        # Construir matriz exp(dt*xi) ~= [[I + dt*hat(w), dt*v], [0, 1]]
        w_dt = w * dt
        theta = np.linalg.norm(w_dt)
        if theta < 1e-9:
            R_inc = np.eye(3)
        else:
            k_hat = np.array([
                [0, -w_dt[2], w_dt[1]],
                [w_dt[2], 0, -w_dt[0]],
                [-w_dt[1], w_dt[0], 0],
            ])
            # Rodrigues
            R_inc = np.eye(3) + np.sin(theta)/theta * k_hat + (1-np.cos(theta))/theta**2 * (k_hat @ k_hat)

        T_inc = np.eye(4)
        T_inc[:3, :3] = R_inc
        T_inc[:3, 3] = v * dt
        T = T @ T_inc
        trajectory.append(T.copy())

    return {
        "trajectory": trajectory,
        "errors_lin_m": errors_lin,
        "errors_ang_rad": errors_ang,
        "n_iters": len(errors_lin),
        "converged": converged_at is not None,
        "converged_at": converged_at,
        "dt": dt,
    }
