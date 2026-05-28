#!/usr/bin/env python3
"""Construye `data/scenes/bin_base.ttt`: escena base para el TFM.

Contenido:
    - Piso (default de CoppeliaSim).
    - UR5 cargado desde la librería de modelos (aliasado /UR5e).
    - Bin (caja hueca de paredes finas).
    - Vision sensor RGB cenital (/rgb_camera).
    - Vision sensor depth cenital (/depth_camera).
    - Luz omnidireccional (/Light).
    - 5 objetos surtidos: 3 cubos RGB + 2 cilindros (/object_1 ... /object_5).

Uso:
    1. Abrir CoppeliaSim Edu V4.10 (open -a CoppeliaSim_Edu).
    2. python scripts/build_bin_base_scene.py
    3. Verificar visualmente en la GUI.
    4. La escena queda guardada en data/scenes/bin_base.ttt.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

UR5_MODEL = "/Applications/CoppeliaSim_Edu.app/Contents/Resources/models/robots/non-mobile/UR5.ttm"
SCENE_OUT = REPO_ROOT / "data" / "scenes" / "bin_base.ttt"


def main() -> int:
    SCENE_OUT.parent.mkdir(parents=True, exist_ok=True)

    if not Path(UR5_MODEL).exists():
        print(f"[FAIL] no encontré UR5.ttm en {UR5_MODEL}")
        print("       verificá la instalación de CoppeliaSim Edu V4.10")
        return 1

    with CoppeliaSimBridge() as bridge:
        sim = bridge.sim
        print("[INFO] limpiando escena actual")
        try:
            sim.closeScene()
        except Exception:
            pass

        # 1. Piso ya existe por defecto en escenas nuevas. Lo dejamos.

        # 2. Cargar UR5
        print(f"[INFO] cargando UR5 desde {UR5_MODEL}")
        ur5_handle = sim.loadModel(UR5_MODEL)
        sim.setObjectAlias(ur5_handle, "UR5e")
        sim.setObjectPosition(ur5_handle, -1, [0.0, 0.0, 0.0])

        # 3. Bin: 4 paredes + piso fino, ~30x30x10 cm
        print("[INFO] construyendo bin")
        bin_pos = [0.5, 0.0, 0.05]  # 50cm en frente del robot, base a 5cm
        wall_t = 0.005
        bin_size = 0.30
        bin_h = 0.10

        # Piso del bin
        floor = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [bin_size, bin_size, wall_t], 0)
        sim.setObjectPosition(floor, -1, [bin_pos[0], bin_pos[1], bin_pos[2] - bin_h / 2])
        sim.setObjectAlias(floor, "bin_floor")
        sim.setShapeColor(floor, None, sim.colorcomponent_ambient_diffuse, [0.4, 0.4, 0.4])

        # 4 paredes
        wall_specs = [
            ("bin_wall_north", [bin_size, wall_t, bin_h], [0, +bin_size / 2, 0]),
            ("bin_wall_south", [bin_size, wall_t, bin_h], [0, -bin_size / 2, 0]),
            ("bin_wall_east",  [wall_t, bin_size, bin_h], [+bin_size / 2, 0, 0]),
            ("bin_wall_west",  [wall_t, bin_size, bin_h], [-bin_size / 2, 0, 0]),
        ]
        wall_handles = []
        for name, size, offset in wall_specs:
            h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
            sim.setObjectPosition(h, -1, [bin_pos[0] + offset[0], bin_pos[1] + offset[1], bin_pos[2]])
            sim.setObjectAlias(h, name)
            sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [0.5, 0.5, 0.5])
            wall_handles.append(h)

        # Agrupar piso + paredes como un solo objeto compuesto y aliasarlo /bin
        # NOTA: groupShapes no siempre está disponible vía ZMQ. Si falla, dejar
        # las piezas sueltas con aliases bin_*; el bridge no requiere /bin
        # estrictamente.
        try:
            bin_group = sim.groupShapes([floor] + wall_handles, False)
            sim.setObjectAlias(bin_group, "bin")
        except Exception as e:
            print(f"[warn] no se pudo agrupar bin ({e}); piezas quedan separadas con aliases bin_*")

        # 4. Vision sensor RGB cenital
        print("[INFO] creando rgb_camera")
        options = 1 + 2  # explicit handling + perspective
        int_params = [640, 480, 0, 0]
        float_params = [
            0.05, 2.0,                         # near, far
            math.radians(60),                  # FOV
            0.1, 0.1, 0.1,                     # cube size
            0.0, 0.0, 0.0, 0.0, 0.0,           # padding
        ]
        rgb_h = sim.createVisionSensor(options, int_params, float_params)
        sim.setObjectAlias(rgb_h, "rgb_camera")
        sim.setObjectPosition(rgb_h, -1, [bin_pos[0], bin_pos[1], 1.0])  # 1m sobre el bin
        sim.setObjectOrientation(rgb_h, -1, [math.pi, 0.0, 0.0])  # mirando hacia abajo

        # 5. Vision sensor depth co-localizado
        depth_h = sim.createVisionSensor(options, int_params, float_params)
        sim.setObjectAlias(depth_h, "depth_camera")
        sim.setObjectPosition(depth_h, -1, [bin_pos[0], bin_pos[1], 1.0])
        sim.setObjectOrientation(depth_h, -1, [math.pi, 0.0, 0.0])

        # 6. Luz omnidireccional
        # NOTA: la API ZMQ de CoppeliaSim no expone addLight/createLight; las
        # luces sólo se crean desde GUI o cargando modelos. La escena default
        # trae 4 luces bajo /DefaultLights (LightA..LightD). Reutilizamos la
        # primera, la reposicionamos sobre el bin y la aliaseamos como /Light
        # para satisfacer la convención de handles del bridge.
        print("[INFO] preparando Light (reutilizando luz default)")
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
            print(f"[warn] no pude aliasear luz default ({e}); se omite /Light")

        # 7. 5 objetos surtidos
        print("[INFO] creando 5 objetos surtidos")
        cube_size = [0.05, 0.05, 0.05]
        cyl_size = [0.05, 0.05, 0.06]
        objs_spec = [
            ("object_1", "cuboid",  cube_size, [0.9, 0.1, 0.1], [-0.08,  0.05, 0.05]),
            ("object_2", "cuboid",  cube_size, [0.1, 0.9, 0.1], [+0.08, -0.05, 0.05]),
            ("object_3", "cuboid",  cube_size, [0.1, 0.1, 0.9], [+0.05, +0.08, 0.05]),
            ("object_4", "cylinder", cyl_size, [0.7, 0.7, 0.7], [-0.05, -0.07, 0.05]),
            ("object_5", "cylinder", cyl_size, [0.9, 0.8, 0.1], [+0.00,  0.00, 0.05]),
        ]
        for alias, shape, size, color, offset in objs_spec:
            ptype = sim.primitiveshape_cuboid if shape == "cuboid" else sim.primitiveshape_cylinder
            h = sim.createPrimitiveShape(ptype, size, 0)
            sim.setObjectAlias(h, alias)
            sim.setObjectPosition(h, -1, [bin_pos[0] + offset[0], bin_pos[1] + offset[1], bin_pos[2] + offset[2]])
            sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
            # Marcar como dinámico para que caigan al iniciar la simulación
            sim.setObjectInt32Param(h, sim.shapeintparam_static, 0)
            sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)

        # 8. Guardar escena
        print(f"[INFO] guardando escena en {SCENE_OUT}")
        sim.saveScene(str(SCENE_OUT))

    print(f"[OK]   escena guardada: {SCENE_OUT}")
    print(f"       tamaño: {SCENE_OUT.stat().st_size / 1024:.1f} KB")
    print()
    print("Próximo paso: abrir CoppeliaSim, File > Open scene...,")
    print(f"             y verificar visualmente que la escena tiene UR5 + bin + 5 objetos + cámaras.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
