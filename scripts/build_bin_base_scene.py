#!/usr/bin/env python3
"""Construye data/scenes/bin_base.ttt — escena completa para bin-picking.

Cambios vs versión anterior:
- Aliasea los 6 joints del UR5 a nombres convencionales
  (shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
   wrist_1_joint, wrist_2_joint, wrist_3_joint).
- Attachea gripper RG2 al force sensor /UR5e/connection y lo aliasea /gripper.
- Crea dummy /tip en el TCP del gripper.
- Bin + cámara + luz + 5 objetos alineados con el alcance del UR5.

Uso:
    1. Abrir CoppeliaSim Edu V4.10.
    2. python scripts/build_bin_base_scene.py
    3. Verificar visualmente.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

UR5_MODEL = "/Applications/CoppeliaSim_Edu.app/Contents/Resources/models/robots/non-mobile/UR5.ttm"
GRIPPER_MODEL = "/Applications/CoppeliaSim_Edu.app/Contents/Resources/models/components/grippers/RG2.ttm"
SCENE_OUT = REPO_ROOT / "data" / "scenes" / "bin_base.ttt"

# Joint names en el orden DFS del kinematic chain del UR5
UR5_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def alias_ur5_joints(sim, ur5_handle) -> list[int]:
    """Encuentra los 6 joints del UR5 en orden DFS y los aliasea a los
    nombres convencionales. Devuelve la lista de handles."""
    descendants = sim.getObjectsInTree(ur5_handle)
    joints_in_order = [
        h for h in descendants if sim.getObjectType(h) == sim.object_joint_type
    ]
    # getObjectsInTree devuelve en orden DFS por defecto.
    assert len(joints_in_order) == 6, (
        f"UR5 debería tener 6 joints, encontré {len(joints_in_order)}"
    )
    for handle, name in zip(joints_in_order, UR5_JOINT_NAMES):
        sim.setObjectAlias(handle, name)
        print(f"  alias  handle={handle} → /{name}")
    return joints_in_order


def find_connection_point(sim, ur5_handle) -> int:
    """Encuentra el punto de conexión (TCP/attach point) del UR5.

    El UR5.ttm del CoppeliaSim Edu V4.10 expone `/UR5/connection` como un
    force sensor (no un dummy), por eso recorremos todos los descendientes
    y buscamos por alias 'connection' independientemente del tipo.
    """
    descendants = sim.getObjectsInTree(ur5_handle)
    for h in descendants:
        try:
            alias = sim.getObjectAlias(h)
            if alias == "connection":
                return h
        except Exception:
            continue
    raise RuntimeError("no encontré objeto /connection en el UR5")


def attach_gripper(sim, connection_handle: int) -> tuple[int, int]:
    """Carga RG2, lo emparenta al force sensor de conexión del UR5, lo
    aliasea como /gripper y crea un dummy /tip en su TCP. Devuelve
    (gripper_handle, tip_handle)."""
    print(f"[INFO] cargando gripper RG2 desde {GRIPPER_MODEL}")
    gripper_handle = sim.loadModel(GRIPPER_MODEL)

    # Posicionar el gripper sobre el connection point y emparentarlo.
    # Usamos keepInPlace=False y reposicionamos explícitamente: así nos
    # aseguramos de que la pose del gripper coincida con la del connection,
    # lo cual es robusto incluso si el dummy ya tiene rotación.
    sim.setObjectParent(gripper_handle, connection_handle, False)
    sim.setObjectPosition(gripper_handle, connection_handle, [0.0, 0.0, 0.0])
    sim.setObjectQuaternion(gripper_handle, connection_handle, [0.0, 0.0, 0.0, 1.0])
    sim.setObjectAlias(gripper_handle, "gripper")
    print(
        f"  gripper attached al connection (handle {connection_handle}), "
        f"alias /gripper (handle {gripper_handle})"
    )

    # Crear un dummy /tip en el TCP. RG2 con dedos extendidos: ~13 cm
    # desde el origen del modelo hasta la punta de los dedos.
    tip_handle = sim.createDummy(0.01)  # 1 cm tamaño visual
    sim.setObjectParent(tip_handle, gripper_handle, False)
    sim.setObjectPosition(tip_handle, gripper_handle, [0.0, 0.0, 0.13])
    sim.setObjectAlias(tip_handle, "tip")
    print(f"  tip creado en TCP (handle {tip_handle})")

    return gripper_handle, tip_handle


def main() -> int:
    SCENE_OUT.parent.mkdir(parents=True, exist_ok=True)
    if not Path(UR5_MODEL).exists():
        print(f"[FAIL] UR5.ttm no encontrado en {UR5_MODEL}")
        return 1
    if not Path(GRIPPER_MODEL).exists():
        print(f"[FAIL] RG2.ttm no encontrado en {GRIPPER_MODEL}")
        return 1

    with CoppeliaSimBridge() as bridge:
        sim = bridge.sim
        print("[INFO] limpiando escena actual")
        try:
            sim.closeScene()
        except Exception:
            pass

        # 1. UR5 con joints aliaseados.
        # UR5.ttm tiene "forward" = -Y en su frame local (verificado empíricamente).
        # Para que el robot mire hacia el bin en +X, rotamos la base +π/2 en Z
        # (yaw = +90°), que mapea -Y → +X.
        print("[INFO] cargando UR5 (rotado +π/2 para que mire a +X)")
        ur5_handle = sim.loadModel(UR5_MODEL)
        sim.setObjectAlias(ur5_handle, "UR5e")
        sim.setObjectPosition(ur5_handle, -1, [0.0, 0.0, 0.0])
        sim.setObjectOrientation(ur5_handle, -1, [0.0, 0.0, math.pi / 2])

        print("[INFO] aliaseando joints del UR5")
        joint_handles = alias_ur5_joints(sim, ur5_handle)

        # 1b. Joints en modo dinámico + motor + position control para que
        # setJointTargetPosition desde Python funcione.
        # NOTA: el UR5.ttm trae un script threaded (/UR5e/Script) que controla
        # los joints en callback mode (8) y sobrescribe nuestros target desde
        # Python. Lo disabled aquí.
        print("[INFO] joints → dynamic + motor + position control")
        for h in joint_handles:
            sim.setJointMode(h, sim.jointmode_dynamic, 0)
            sim.setObjectInt32Param(h, sim.jointintparam_motor_enabled, 1)
            sim.setObjectInt32Param(h, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
        try:
            scr = sim.getObject("/UR5e/Script")
            sim.setObjectInt32Param(scr, sim.scriptintparam_enabled, 0)
            print("[INFO] UR5 threaded script disabled")
        except Exception as e:
            print(f"[warn] disable UR5 script: {e}")

        # 2. Gripper + tip
        connection_handle = find_connection_point(sim, ur5_handle)
        gripper_handle, tip_handle = attach_gripper(sim, connection_handle)

        # 3. Bin alineado con el alcance del UR5
        print("[INFO] construyendo bin")
        # Bin a 28 cm: con paredes BAJAS (bin_h=0.03) los cubos descansan en
        # z=0.29 (top de bin floor + cube_h/2) y las paredes no interfieren
        # con el descenso del brazo del UR5. TCP descend del UR5 llega a
        # z=0.317 — justo encima de los cubos para que el RG2 los agarre.
        bin_pos = [0.46, -0.10, 0.28]
        wall_t = 0.005
        bin_size = 0.30
        # Paredes BAJAS (3 cm) para que el brazo del UR5 pueda descender sin
        # colisionar lateralmente. Lo importante es que los cubos no se
        # desparramen, no que sea un bin profundo.
        bin_h = 0.03

        floor = sim.createPrimitiveShape(
            sim.primitiveshape_cuboid, [bin_size, bin_size, wall_t], 0
        )
        sim.setObjectPosition(
            floor, -1, [bin_pos[0], bin_pos[1], bin_pos[2] - bin_h / 2]
        )
        sim.setObjectAlias(floor, "bin_floor")
        sim.setShapeColor(
            floor, None, sim.colorcomponent_ambient_diffuse, [0.4, 0.4, 0.4]
        )

        wall_specs = [
            ("bin_wall_north", [bin_size, wall_t, bin_h], [0, +bin_size / 2, 0]),
            ("bin_wall_south", [bin_size, wall_t, bin_h], [0, -bin_size / 2, 0]),
            ("bin_wall_east",  [wall_t, bin_size, bin_h], [+bin_size / 2, 0, 0]),
            ("bin_wall_west",  [wall_t, bin_size, bin_h], [-bin_size / 2, 0, 0]),
        ]
        wall_handles = []
        for name, size, offset in wall_specs:
            h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
            sim.setObjectPosition(
                h, -1, [bin_pos[0] + offset[0], bin_pos[1] + offset[1], bin_pos[2]]
            )
            sim.setObjectAlias(h, name)
            sim.setShapeColor(
                h, None, sim.colorcomponent_ambient_diffuse, [0.5, 0.5, 0.5]
            )
            wall_handles.append(h)

        try:
            bin_group = sim.groupShapes([floor] + wall_handles, False)
            sim.setObjectAlias(bin_group, "bin")
            # CRÍTICO: groupShapes preserva static=1 (bien) pero pone
            # respondable=0 por default. Sin esto el bin es un "fantasma" y
            # los cubos lo atraviesan al iniciar simulación, cayendo al piso.
            sim.setObjectInt32Param(bin_group, sim.shapeintparam_static, 1)
            sim.setObjectInt32Param(bin_group, sim.shapeintparam_respondable, 1)
            print("[INFO] /bin: static=1, respondable=1 (sólido para colisiones)")
        except Exception as e:
            print(f"[warn] no se pudo agrupar bin ({e}); piezas quedan separadas")

        # 4. Vision sensors cenitales sobre el bin (1 m de altura)
        print("[INFO] creando rgb_camera y depth_camera")
        options = 1 + 2  # explicit handling + perspective
        int_params = [640, 480, 0, 0]
        float_params = [
            0.05, 2.0,                # near, far
            math.radians(60),         # FOV
            0.1, 0.1, 0.1,            # cube size (visual)
            0.0, 0.0, 0.0, 0.0, 0.0,  # padding
        ]

        # Cámara cenital sobre el bin (1.2 m altura — alta para que entren
        # robot + bin en el frame).
        rgb_h = sim.createVisionSensor(options, int_params, float_params)
        sim.setObjectAlias(rgb_h, "rgb_camera")
        sim.setObjectPosition(rgb_h, -1, [bin_pos[0] - 0.20, bin_pos[1] + 0.20, 1.30])
        sim.setObjectOrientation(rgb_h, -1, [math.pi, 0.0, 0.0])

        depth_h = sim.createVisionSensor(options, int_params, float_params)
        sim.setObjectAlias(depth_h, "depth_camera")
        sim.setObjectPosition(depth_h, -1, [bin_pos[0] - 0.20, bin_pos[1] + 0.20, 1.30])
        sim.setObjectOrientation(depth_h, -1, [math.pi, 0.0, 0.0])

        # 5. Luz: reutilizar default (CoppeliaSim no expone addLight vía ZMQ)
        print("[INFO] aliaseando luz default como /Light")
        try:
            all_handles = sim.getObjectsInTree(sim.handle_scene)
            light_h = None
            for h in all_handles:
                if sim.getObjectType(h) == sim.sceneobject_light:
                    light_h = h
                    break
            if light_h is None:
                print("[warn] no encontré ninguna luz en la escena; se omite /Light")
            else:
                sim.setObjectAlias(light_h, "Light")
                sim.setObjectPosition(light_h, -1, [bin_pos[0], bin_pos[1], 1.2])
        except Exception as e:
            print(f"[warn] no pude aliasear luz default: {e}")

        # 6. 5 objetos en el bin
        print("[INFO] creando 5 objetos surtidos")
        cube_size = [0.05, 0.05, 0.05]
        cyl_size = [0.05, 0.05, 0.06]
        # Cubos descansando sobre el piso del bin (z = -bin_h/2 + cube_h/2)
        # object_1 (rojo) en el centro = alineado con TCP descend del UR5.
        # Z offset desde bin_pos: -bin_h/2 + cube_size[2]/2 = -0.075 + 0.025 = -0.05
        z_floor = -bin_h / 2 + cube_size[2] / 2
        objs_spec = [
            ("object_1", "cuboid",   cube_size, [0.9, 0.1, 0.1], [+0.00,  0.00, z_floor]),
            ("object_2", "cuboid",   cube_size, [0.1, 0.9, 0.1], [+0.08, +0.07, z_floor]),
            ("object_3", "cuboid",   cube_size, [0.1, 0.1, 0.9], [-0.08, +0.08, z_floor]),
            ("object_4", "cylinder", cyl_size,  [0.7, 0.7, 0.7], [+0.08, -0.07, z_floor]),
            ("object_5", "cylinder", cyl_size,  [0.9, 0.8, 0.1], [-0.08, -0.07, z_floor]),
        ]
        for alias, shape, size, color, offset in objs_spec:
            ptype = (
                sim.primitiveshape_cuboid if shape == "cuboid"
                else sim.primitiveshape_cylinder
            )
            h = sim.createPrimitiveShape(ptype, size, 0)
            sim.setObjectAlias(h, alias)
            sim.setObjectPosition(
                h,
                -1,
                [
                    bin_pos[0] + offset[0],
                    bin_pos[1] + offset[1],
                    bin_pos[2] + offset[2],
                ],
            )
            sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
            # Dinámicos + responsable a colisiones: el gripper puede agarrarlos.
            sim.setObjectInt32Param(h, sim.shapeintparam_static, 0)
            sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)

        # 7. Guardar escena
        print(f"[INFO] guardando escena en {SCENE_OUT}")
        sim.saveScene(str(SCENE_OUT))

    print(f"[OK]   escena guardada: {SCENE_OUT}")
    print(f"       tamaño: {SCENE_OUT.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
