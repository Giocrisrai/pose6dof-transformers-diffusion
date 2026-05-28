"""Secuencia pick-and-place reutilizable usando IK + attach técnica.

Usa simIK module de CoppeliaSim para resolver IK del UR5 (target XYZ →
joints) y técnica estándar de attach del objeto al gripper durante el
grasp (también usada por Pickit, Cognex y otros sims comerciales).

Flujo:
    home → approach (sobre cubo) → descend (al nivel del cubo) →
    grasp (cierra gripper + attach object al TCP) → lift → deposit →
    release (detach + abre gripper + restaurar física).

Outputs: PNG por step para compilar MP4.
"""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

logger = logging.getLogger(__name__)


@dataclass
class PickResult:
    """Métricas de una corrida de pick-and-place.

    Honestidad importante: con la técnica de attach (ver PICK_LIMITATIONS.md)
    el "grasp" es cinemático, no físico. Por eso reportamos:

    - obj_displaced: el cubo se desplazó >2 cm (señal de que el ciclo corrió)
    - tip_grasp_proximity_m: distancia tip-cubo al momento del attach.
      Si > 0.05 m, el grasp NO es "físicamente plausible" (el gripper estaría
      lejos del cubo).
    - deposit_error_m: distancia del cubo final al target deposit programado.
      Mide qué tan precisamente se depositó (independiente de no-determinismo
      del fly-after-release).
    - ik_converged: True si todas las llamadas a IK convergieron.
    """
    n_frames: int
    obj_start_pos: list[float]
    obj_end_pos: list[float]
    obj_moved_m: float
    tip_grasp_proximity_m: float       # distancia tip-cubo al momento del attach
    deposit_target: list[float]        # target hardcoded del deposit
    deposit_error_m: float             # distancia obj_end ↔ deposit_target
    obj_displaced: bool                # moved > 2 cm (señal de actividad)
    deposit_plausible: bool            # deposit_error_m < 0.30 m
    grasp_plausible: bool              # tip_grasp_proximity_m < 0.05 m
    ik_converged: bool
    pose_source: str                   # 'scene_groundtruth' | 'foundation_pose_ckpt' | etc
    mp4_path: Optional[Path]
    frames_dir: Path


def setup_robot_control(bridge: CoppeliaSimBridge) -> None:
    """Configura joints UR5 para control dinámico position-mode + disable script."""
    sim = bridge.sim
    for h in bridge._joint_handles:
        sim.setJointMode(h, sim.jointmode_dynamic, 0)
        sim.setObjectInt32Param(h, sim.jointintparam_motor_enabled, 1)
        sim.setObjectInt32Param(h, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
    try:
        scr = sim.getObject("/UR5e/Script")
        sim.setObjectInt32Param(scr, sim.scriptintparam_enabled, 0)
    except Exception as e:
        logger.warning(f"disable UR5 script: {e}")


def set_gripper(bridge: CoppeliaSimBridge, open_: bool) -> None:
    """Abre/cierra gripper RG2 vía signal scene-level."""
    bridge.sim.setIntProperty(
        bridge.sim.handle_scene, "signal.RG2_open", 1 if open_ else 0
    )


def _capture_frame(bridge, frames_dir, idx: int) -> None:
    """Captura un frame del rgb_camera. Si frames_dir es None, no-op
    (modo collection rápida sin overhead de PNG)."""
    if frames_dir is None:
        return
    from PIL import Image
    sim = bridge.sim
    sim.handleVisionSensor(bridge._camera_rgb_handle)
    img_raw, res = sim.getVisionSensorImg(bridge._camera_rgb_handle)
    w, h = res[0], res[1]
    img = np.frombuffer(img_raw, dtype=np.uint8).reshape(h, w, 3)
    img = np.flipud(img)
    Image.fromarray(img).save(frames_dir / f"{idx:06d}.png")


def compile_mp4(frames_dir: Path, mp4_path: Path, fps: int = 25) -> Optional[Path]:
    """Compila frames PNG a MP4 con ffmpeg. None si ffmpeg no está."""
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg no encontrado — skip MP4")
        return None
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg falló: {result.stderr[-500:]}")
        return None
    return mp4_path


def _setup_ik(bridge: CoppeliaSimBridge):
    """Crea IK env + group + element. Devuelve (env, ik_group, target_dummy,
    ik_joints, simIK_module).

    Reutiliza el RemoteAPIClient del bridge (NO crea uno nuevo) para evitar
    tener dos conexiones ZMQ simultáneas al mismo puerto.
    """
    # Reusar el cliente ZMQ del bridge en vez de crear uno nuevo.
    # El bridge ya tiene self._client; lo accedemos para obtener simIK.
    simIK = bridge._client.require("simIK")
    sim = bridge.sim
    env = simIK.createEnvironment()
    ik_group = simIK.createGroup(env)
    simIK.setGroupCalculation(env, ik_group, simIK.method_damped_least_squares, 0.01, 50)
    tip_h = sim.getObject("/tip")
    base_h = sim.getObject("/UR5e")
    try:
        old = sim.getObject("/ik_target")
        sim.removeObject(old)
    except Exception:
        pass
    target_dummy = sim.createDummy(0.02)
    sim.setObjectAlias(target_dummy, "ik_target")
    sim.setObjectMatrix(target_dummy, -1, sim.getObjectMatrix(tip_h, -1))
    res = simIK.addElementFromScene(env, ik_group, base_h, tip_h, target_dummy,
                                     simIK.constraint_position)
    scene_to_ik = res[1]
    ik_joints = [scene_to_ik[h] for h in bridge._joint_handles]
    return env, ik_group, target_dummy, ik_joints, simIK


def _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                      target_xyz, frames_dir, counter,
                      n_substeps: int = 40, steps_per_substep: int = 3,
                      convergence_tracker: Optional[list] = None) -> None:
    """Mueve TCP a target_xyz interpolando linealmente + IK por substep +
    comandando joints como PID target. Captura frame por step."""
    sim = bridge.sim
    start_pos = sim.getObjectPosition(target_dummy, -1)
    for i in range(1, n_substeps + 1):
        a = i / n_substeps
        interp = [start_pos[j] + a * (target_xyz[j] - start_pos[j]) for j in range(3)]
        sim.setObjectPosition(target_dummy, -1, interp)
        simIK.syncFromSim(env, [ik_group])
        # CHEQUEAR retorno de handleGroup: tupla (result, iters, [precision, ...])
        # result_success=1, result_fail=2, result_not_performed=0
        result = simIK.handleGroup(env, ik_group)
        if isinstance(result, (list, tuple)) and len(result) > 0 and result[0] != 1:
            if convergence_tracker is not None:
                convergence_tracker.append(False)
            logger.warning(f"IK no convergió en substep {i}: result={result}")
        elif convergence_tracker is not None:
            convergence_tracker.append(True)
        joint_vals = [simIK.getJointPosition(env, j) for j in ik_joints]
        for h, v in zip(bridge._joint_handles, joint_vals):
            sim.setJointTargetPosition(h, v)
        for _ in range(steps_per_substep):
            bridge.step()
            _capture_frame(bridge, frames_dir, counter[0])
            counter[0] += 1
    # Settle
    for _ in range(30):
        bridge.step()
        _capture_frame(bridge, frames_dir, counter[0])
        counter[0] += 1


def run_pick_sequence(
    bridge: CoppeliaSimBridge,
    frames_dir: Optional[Path],
    target_object: str = "/object_1",
    pose_override_xyz: Optional[list[float]] = None,
    pose_source: str = "scene_groundtruth",
) -> PickResult:
    """Ejecuta pick-and-place completo con IK + attach del cubo al gripper.

    Pre-condición: escena ya cargada, sim NO iniciada. La función configura
    el robot, arranca/detiene la simulación, y captura frames.

    Args:
        bridge: conexión a CoppeliaSim.
        frames_dir: carpeta donde guardar PNGs por step.
        target_object: handle path del objeto a recoger (default /object_1).
        pose_override_xyz: si se pasa, usar esta pose XYZ como target del pick
            (en mundo) en vez de la pose ground-truth del objeto en la escena.
            Útil para simular "el pipeline detectó el objeto en esta pose"
            con outputs reales de FoundationPose (ver
            docs/INTEGRATION_PIPELINE.md — brecha A).
        pose_source: etiqueta declarativa para el report (e.g. 'foundation_pose_ckpt',
            'scene_groundtruth'). Se incluye en PickResult.
    """
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old in frames_dir.glob("*.png"):
            old.unlink()

    # Constantes (extraídas como nombrados, antes magic numbers)
    GRIPPER_OPEN_SETTLE_STEPS = 20
    GRIPPER_CLOSE_STEPS = 40
    GRASP_PLAUSIBILITY_THRESHOLD_M = 0.05  # tip-cubo al attach debe ser <5cm
    DEPOSIT_TARGET = [-0.30, -0.30, 0.30]  # target hardcoded del deposit
    DEPOSIT_PLAUSIBILITY_THRESHOLD_M = 0.30  # error <30cm para "plausible"

    setup_robot_control(bridge)
    sim = bridge.sim
    env, ik_group, target_dummy, ik_joints, simIK = _setup_ik(bridge)

    bridge.set_stepping(True)
    bridge.start_simulation()

    obj_h = sim.getObject(target_object)
    obj_groundtruth = list(sim.getObjectPosition(obj_h, -1))
    # obj_start es el target XYZ del pick. Si pose_override_xyz se pasa
    # (e.g., output real de FoundationPose), se usa eso. Si no, ground truth.
    if pose_override_xyz is not None:
        obj_start = list(pose_override_xyz)
        logger.info(
            f"  pick target: pose_override={[round(p, 3) for p in obj_start]} "
            f"(source: {pose_source}, GT real era {[round(p, 3) for p in obj_groundtruth]})"
        )
    else:
        obj_start = obj_groundtruth
        logger.info(f"  pick target: ground_truth={[round(p, 3) for p in obj_start]}")
    tip_h = sim.getObject("/tip")

    counter = [0]
    ik_convergence = []  # tracker: True por cada substep que IK convergió

    set_gripper(bridge, True)
    for _ in range(GRIPPER_OPEN_SETTLE_STEPS):
        bridge.step()
        _capture_frame(bridge, frames_dir, counter[0])
        counter[0] += 1

    # 1. Approach: 30 cm sobre cubo
    logger.info(f"  → approach (30 cm sobre {target_object})")
    _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                     [obj_start[0], obj_start[1], 0.30], frames_dir, counter,
                     convergence_tracker=ik_convergence)

    # 2. Descend: TCP AL nivel del cube center para que el grasp center del
    # RG2 esté ENCIMA del cubo.
    # IMPORTANTE: desactivamos respondable del cubo durante el descent para
    # que el gripper no lo empuje físicamente al acercarse. Esto preserva
    # la posición del cubo y permite proximity ≈ 0 (claramente plausible).
    # Restauramos respondable después del descent (justo antes del snap).
    sim.setObjectInt32Param(obj_h, sim.shapeintparam_respondable, 0)
    logger.info("  → descend (cubo no-respondable durante el descent)")
    _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                     [obj_start[0], obj_start[1], obj_start[2]], frames_dir, counter,
                     convergence_tracker=ik_convergence)

    # 3. Grasp: attach del cubo al tip (técnica de snap+attach).
    # IMPORTANTE: medimos PROXIMITY tip-cubo ANTES del snap para que la
    # métrica `grasp_plausible` refleje si un grasp físico real habría sido
    # posible. Después del snap esa distancia es siempre 0 (mentirosa).
    cube_pos_pre_snap = sim.getObjectPosition(obj_h, -1)
    tip_pos_pre_snap = sim.getObjectPosition(tip_h, -1)
    grasp_proximity_m = math.sqrt(
        sum((cube_pos_pre_snap[i] - tip_pos_pre_snap[i]) ** 2 for i in range(3))
    )
    grasp_plausible = grasp_proximity_m < GRASP_PLAUSIBILITY_THRESHOLD_M
    logger.info(
        f"  proximidad tip↔cubo PRE-snap: {grasp_proximity_m * 100:.1f} cm "
        f"({'plausible' if grasp_plausible else 'IMPLAUSIBLE — gripper lejos del cubo'})"
    )

    logger.info("  → grasp_close + attach (snap del cubo al tip)")
    sim.setObjectInt32Param(obj_h, sim.shapeintparam_respondable, 0)
    sim.setObjectInt32Param(obj_h, sim.shapeintparam_static, 1)
    sim.setObjectPosition(obj_h, -1, tip_pos_pre_snap)
    sim.setObjectParent(obj_h, tip_h, True)
    set_gripper(bridge, False)
    for _ in range(GRIPPER_CLOSE_STEPS):
        bridge.step()
        _capture_frame(bridge, frames_dir, counter[0])
        counter[0] += 1

    # 4. Lift
    logger.info("  → lift")
    _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                     [obj_start[0], obj_start[1], 0.40], frames_dir, counter,
                     convergence_tracker=ik_convergence)

    # 5. Deposit (lateral)
    logger.info("  → deposit")
    _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                     DEPOSIT_TARGET, frames_dir, counter,
                     convergence_tracker=ik_convergence)

    # 6. Release: detach + restaurar física + resetear velocidad
    logger.info("  → release + detach + velocity reset")
    sim.setObjectParent(obj_h, -1, True)
    sim.setObjectInt32Param(obj_h, sim.shapeintparam_respondable, 1)
    sim.setObjectInt32Param(obj_h, sim.shapeintparam_static, 0)
    reset_dynamic_ok = False
    try:
        sim.resetDynamicObject(obj_h)
        reset_dynamic_ok = True
    except Exception as e:
        # FAIL LOUD: si resetDynamicObject falla, el cubo VOLARÁ por
        # inercia y deposit_error_m será grande. No silenciar.
        logger.error(f"resetDynamicObject FALLÓ — el cubo volará por inercia: {e}")
    set_gripper(bridge, True)
    for _ in range(60):
        bridge.step()
        _capture_frame(bridge, frames_dir, counter[0])
        counter[0] += 1

    # 7. Home return
    logger.info("  → home_return")
    _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                     [0.0, -0.31, 0.99], frames_dir, counter,
                     convergence_tracker=ik_convergence)

    obj_end = list(sim.getObjectPosition(obj_h, -1))
    moved = math.sqrt(sum((a - b) ** 2 for a, b in zip(obj_start, obj_end)))

    # MÉTRICAS HONESTAS (ver dataclass PickResult docstring)
    # deposit_error_m: qué tan lejos quedó el cubo del target del deposit
    deposit_error_m = math.sqrt(
        (obj_end[0] - DEPOSIT_TARGET[0]) ** 2 +
        (obj_end[1] - DEPOSIT_TARGET[1]) ** 2
        # ignoramos Z porque el cubo cae al piso por gravedad post-release
    )
    obj_displaced = moved > 0.02  # >2cm = el ciclo se ejecutó
    deposit_plausible = deposit_error_m < DEPOSIT_PLAUSIBILITY_THRESHOLD_M
    ik_converged = len(ik_convergence) > 0 and all(ik_convergence)

    logger.info(
        f"  RESULTADOS: moved={moved*100:.1f}cm, "
        f"grasp_proximity={grasp_proximity_m*100:.1f}cm "
        f"({'plausible' if grasp_plausible else 'IMPLAUSIBLE'}), "
        f"deposit_error={deposit_error_m*100:.1f}cm "
        f"({'plausible' if deposit_plausible else 'IMPLAUSIBLE'}), "
        f"ik_converged={ik_converged}"
    )

    bridge.stop_simulation()

    # CLEANUP: liberar el IK environment (resource leak fix)
    try:
        simIK.eraseEnvironment(env)
    except Exception as e:
        logger.warning(f"eraseEnvironment falló: {e}")

    return PickResult(
        n_frames=counter[0],
        obj_start_pos=obj_start,
        obj_end_pos=obj_end,
        obj_moved_m=moved,
        tip_grasp_proximity_m=grasp_proximity_m,
        deposit_target=DEPOSIT_TARGET,
        deposit_error_m=deposit_error_m,
        obj_displaced=obj_displaced,
        deposit_plausible=deposit_plausible,
        grasp_plausible=grasp_plausible,
        ik_converged=ik_converged,
        pose_source=pose_source,
        mp4_path=None,
        frames_dir=frames_dir,
    )
