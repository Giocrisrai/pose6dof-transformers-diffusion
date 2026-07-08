"""Render cinematográfico del pick de la pieza text-to-CAD desde dentro de
CoppeliaSim (3ª persona) + HUD superpuesto. Interfaz visual de nuestra simulación."""
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm"); sys.path.insert(0, str(REPO))
from src.simulation.cine_camera import look_at_matrix, orbit_position
from src.simulation.coppeliasim_bridge import CameraConfig, CoppeliaSimBridge
from src.simulation.pick_sequence import run_pick_sequence

EXP = REPO/"experiments/results/exp27_text_to_cad"
OBJ = str(EXP/"assets/test_bracket.obj")
FRAMES = EXP/"cine_frames"
RES = (1280, 720)

with CoppeliaSimBridge() as bridge:
    bridge.set_stepping(True)
    bridge.load_scene(REPO/"data/scenes/bin_base.ttt")
    sim = bridge.sim
    # aparcar objetos por defecto
    for i in range(1,6):
        try:
            h=sim.getObject(f"/object_{i}"); sim.setObjectInt32Param(h,sim.shapeintparam_static,1)
            sim.setObjectPosition(h,-1,[5.,5.,0.05+0.1*i])
        except Exception: pass
    # importar el bracket en la zona diestra, apoyado
    obj = sim.importShape(0, OBJ, 0, 0.0001, 0.001)
    sim.setObjectAlias(obj, "bracket_t2c")
    sim.setObjectInt32Param(obj, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(obj, sim.shapeintparam_respondable, 1)
    sim.computeMassAndInertia(obj, 2700)
    sim.setObjectColor(obj, 0, sim.colorcomponent_ambient_diffuse, [0.85,0.15,0.10])
    sim.setObjectPosition(obj, -1, [-0.05,-0.22,0.05])
    sim.setObjectOrientation(obj, -1, [0.0,0.0,0.4])

    # cámara cinematográfica 3ª persona (encuadre validado)
    cine = sim.createVisionSensor(1, [RES[0],RES[1],0,0],
                                  [0.01, 8.0, math.radians(56), 0.05,0.05,0.05, 0.6,0.6,0.6, 0,0])
    sim.setObjectAlias(cine, "cine_cam")
    center = (-0.08, -0.16, 0.10); campos = (0.62, 0.42, 0.78)
    sim.setObjectMatrix(cine, -1, look_at_matrix(campos, center))

    # usar la cine como cámara del bridge para que run_pick_sequence capture desde ella
    bridge._camera_rgb_handle = cine
    bridge._camera_depth_handle = cine
    bridge.camera_config = CameraConfig(resolution=RES)

    result = run_pick_sequence(bridge, FRAMES, target_object="/bracket_t2c", pose_override_xyz=None)
    print(f"pick: proximity={result.tip_grasp_proximity_m*100:.1f}cm plausible={result.grasp_plausible} "
          f"moved={result.obj_moved_m*100:.1f}cm ik={result.ik_converged}")
n = len(list(FRAMES.glob("*.png")))
print("frames cine:", n)
