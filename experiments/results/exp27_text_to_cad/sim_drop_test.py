#!/usr/bin/env python3
"""
exp27 · text-to-CAD — Paso 1: física en CoppeliaSim.

Importa la pieza generada (test_bracket.obj) en una escena con bin y verifica
que cae y se asienta correctamente sobre la base del contenedor, registrando la
trayectoria z(t). Evidencia de que el mesh generado es válido para dinámica.

Requiere CoppeliaSim Edu V4.10 abierto (open -a CoppeliaSim_Edu), servidor ZMQ
en localhost:23000.

Uso:
    python sim_drop_test.py
"""
import time
from pathlib import Path

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

HERE = Path(__file__).resolve().parent
OBJ = str(HERE / "assets" / "test_bracket.obj")
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)

client = RemoteAPIClient(host="localhost", port=23000)
sim = client.getObject("sim")
if sim.getSimulationState() != sim.simulation_stopped:
    sim.stopSimulation(); time.sleep(0.4)
sim.closeScene()

# --- Suelo + bin ---
BIN_X, BIN_Y, BIN_Z, CX = 0.30, 0.30, 0.08, 0.0
floor = sim.createPrimitiveShape(sim.primitiveshape_plane, [1.5, 1.5, 0.005])
sim.setObjectAlias(floor, "Floor"); sim.setObjectPosition(floor, -1, [0, 0, 0])
base = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [BIN_X, BIN_Y, 0.006])
sim.setObjectAlias(base, "Bin_base"); sim.setObjectPosition(base, -1, [CX, 0, 0.003])
for name, size, pos in [
    ("Bin_Xp", [0.008, BIN_Y, BIN_Z], [CX + BIN_X/2, 0, BIN_Z/2]),
    ("Bin_Xm", [0.008, BIN_Y, BIN_Z], [CX - BIN_X/2, 0, BIN_Z/2]),
    ("Bin_Yp", [BIN_X, 0.008, BIN_Z], [CX, BIN_Y/2, BIN_Z/2]),
    ("Bin_Ym", [BIN_X, 0.008, BIN_Z], [CX, -BIN_Y/2, BIN_Z/2]),
]:
    h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size)
    sim.setObjectAlias(h, name); sim.setObjectPosition(h, -1, pos)
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
for h in (base,):
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)

# --- Importar bracket (OBJ, escala mm->m) y soltarlo ---
obj = sim.importShape(0, OBJ, 0, 0.0001, 0.001)
sim.setObjectAlias(obj, "Bracket_text2cad")
sim.setObjectInt32Param(obj, sim.shapeintparam_static, 0)
sim.setObjectInt32Param(obj, sim.shapeintparam_respondable, 1)
sim.computeMassAndInertia(obj, 2700)
sim.setObjectPosition(obj, -1, [CX + 0.02, 0.01, 0.18])
sim.setObjectOrientation(obj, -1, [0.3, 0.2, 0.5])

vs = sim.createVisionSensor(1, [640, 480, 0, 0],
                            [0.01, 3.0, 1.0472, 0.05,0.05,0.05, 0.5,0.5,0.5, 0,0])
sim.setObjectAlias(vs, "topcam"); sim.setObjectPosition(vs, -1, [CX, 0, 0.9])
sim.setObjectOrientation(vs, -1, [np.pi, 0, 0])

# --- Simular y registrar z(t) ---
sim.setStepping(True); sim.startSimulation()
zs, ts = [], []
for _ in range(250):
    sim.step()
    zs.append(sim.getObjectPosition(obj, -1)[2]); ts.append(sim.getSimulationTime())
zs = np.array(zs)

sim.handleVisionSensor(vs)
img, res = sim.getVisionSensorImg(vs)
buf = sim.unpackUInt8Table(img) if isinstance(img, str) else list(img)
from PIL import Image
Image.fromarray(np.flipud(np.array(buf, np.uint8).reshape(res[1], res[0], 3))).save(FIGS / "sim_drop_final.png")
final = sim.getObjectPosition(obj, -1)
sim.stopSimulation()

settled = np.std(zs[-30:]) < 1e-3
ok = (zs[0]-zs.min() > 0.05) and settled and (zs[-1] > -0.02) and abs(final[0]-CX) < BIN_X/2+0.02
print(f"z0={zs[0]*1000:.0f}mm  zmin={zs.min()*1000:.1f}mm  zf={zs[-1]*1000:.1f}mm  "
      f"asentado={settled}  -> {'OK' if ok else 'REVISAR'}")
np.save(FIGS / "sim_z_traj.npy", np.column_stack([ts, zs]))
