"""
E2E real en simulación con la pieza text-to-CAD:
  bin_base.ttt (robot UR5e + IK + gripper + cámara) -> importar bracket ->
  soltar y asentar -> CAPTURAR DEPTH REAL de la cámara -> estimar pose 6-DoF
  (registro model-based) -> ejecutar PICK REAL (IK + snap+attach) con esa pose.

Cierra dos brechas del proxy previo:
  (1) la observación es depth REAL renderizado por CoppeliaSim (no sintético).
  (2) el agarre es el mecanismo REAL del TFM (IK + snap+attach, run_pick_sequence).

El estimador de pose sigue siendo clásico (FoundationPose neuronal necesita
GPU/Colab), pero ahora alimentado con datos de sensor reales.
"""
import sys, json, math
from pathlib import Path
import numpy as np

REPO = Path("/Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm")
sys.path.insert(0, str(REPO))
import open3d as o3d
import trimesh
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.pick_sequence import run_pick_sequence

SCR = Path(__file__).resolve().parent / "figs"
ASSETS = REPO / "experiments/results/exp27_text_to_cad/assets"
OBJ = str(ASSETS / "test_bracket.obj")

# ---------- Modelo CAD para registro ----------
tm = trimesh.load(ASSETS / "test_bracket.stl"); tm.apply_scale(0.001)
VOX = 0.003
def to_pcd(p):
    q = o3d.geometry.PointCloud(); q.points = o3d.utility.Vector3dVector(np.asarray(p)); return q
model = to_pcd(tm.sample(6000)).voxel_down_sample(VOX)
model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*3, max_nn=30))
fpfh = lambda pc: o3d.pipelines.registration.compute_fpfh_feature(
    pc, o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*5, max_nn=100))
model_fpfh = fpfh(model)

def backproject(depth, mask, K, T_wc):
    """depth+mask (H,W) -> nube del objeto en mundo (convención zc=+1 validada)."""
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    d = depth.reshape(-1); u = us.reshape(-1); v = vs.reshape(-1)
    valid = mask.reshape(-1) & (d > 0.06) & (d < 1.95)
    d, u, v = d[valid], u[valid], v[valid]
    u = (W - 1 - u); v = (H - 1 - v)   # convención de imagen del sensor (rot 180°), validada vs GT
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    pc = np.stack([(u-cx)/fx*d, (v-cy)/fy*d, d], axis=1)   # zc=+d
    return (T_wc[:3,:3] @ pc.T).T + T_wc[:3,3]

with CoppeliaSimBridge() as bridge:
    bridge.set_stepping(True)
    bridge.load_scene(REPO / "data/scenes/bin_base.ttt")
    sim = bridge.sim
    cam = bridge._camera_rgb_handle
    near = sim.getObjectFloatParam(cam, sim.visionfloatparam_near_clipping)
    far = sim.getObjectFloatParam(cam, sim.visionfloatparam_far_clipping)

    # Aparcar los objetos por defecto /object_1../object_5 fuera de escena
    for i in range(1, 6):
        try:
            h = sim.getObject(f"/object_{i}")
            sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
            sim.setObjectPosition(h, -1, [5.0, 5.0, 0.05 + 0.1*i])
        except Exception:
            pass

    # Importar el bracket text-to-CAD como objeto objetivo
    obj = sim.importShape(0, OBJ, 0, 0.0001, 0.001)
    sim.setObjectAlias(obj, "bracket_t2c")
    sim.setObjectInt32Param(obj, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(obj, sim.shapeintparam_respondable, 1)
    sim.computeMassAndInertia(obj, 2700)
    sim.setObjectColor(obj, 0, sim.colorcomponent_ambient_diffuse, [0.85, 0.15, 0.10])
    # Soltar sobre el centro del bin (bin_base: bin alrededor de x~0, y~0)
    sim.setObjectPosition(obj, -1, [0.0, 0.0, 0.15])
    sim.setObjectOrientation(obj, -1, [0.25, 0.15, 0.6])

    # ---- Fase A: soltar, asentar y CAPTURAR DEPTH REAL ----
    bridge.start_simulation()
    for _ in range(120):
        bridge.step()
    rgb, depth_bridge = bridge.capture_rgbd()   # <-- datos de sensor REALES
    # el bridge convierte con clips por defecto (0.1,3.0); re-escalar a reales
    depth = near + (depth_bridge - 0.1) / (3.0 - 0.1) * (far - near)
    K = bridge.camera_config.K
    T_wc = bridge.get_camera_pose()
    gt_pose = bridge.get_object_pose("bracket_t2c")
    settled_pos = list(sim.getObjectPosition(obj, -1))
    settled_quat = list(sim.getObjectQuaternion(obj, -1))
    # máscara del objeto por color rojo (= mask del detector que recibe FoundationPose)
    r, g, bl = rgb[:,:,0].astype(int), rgb[:,:,1].astype(int), rgb[:,:,2].astype(int)
    mask = (r > 110) & (g < 90) & (bl < 90)
    print(f"[A] depth real: {depth.shape} rango {depth.min():.2f}-{depth.max():.2f} m | máscara: {mask.sum()} px")
    print(f"[A] GT pose bracket (settled): xyz={np.round(gt_pose[:3,3],3).tolist()}")
    np.save(SCR/"e2e_depth.npy", depth); np.save(SCR/"e2e_rgb.npy", rgb)
    bridge.stop_simulation()

    # ---- Fase B: nube del objeto desde depth+mask reales ----
    gt_xyz = gt_pose[:3,3]
    world = backproject(depth, mask, K, T_wc)
    derr = np.linalg.norm(world.mean(0)[:2] - gt_xyz[:2])
    print(f"[B] nube objeto: {len(world)} pts | centroide={np.round(world.mean(0),3).tolist()} "
          f"err_xy_vs_GT={derr*100:.1f}cm | bbox={np.round(world.max(0)-world.min(0),3).tolist()}")
    assert len(world) > 100 and derr < 0.05, f"máscara/nube dudosa (pts={len(world)}, err={derr*100:.1f}cm)"
    obs = to_pcd(world).voxel_down_sample(VOX)
    obs, _ = obs.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    obs.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*3, max_nn=30))
    print(f"[B] nube de depth real validada (err centroide {derr*100:.1f}cm), {len(obs.points)} pts")

    # ---- Registro model-based (multi-hipótesis + selección por fitness) ----
    obs_fpfh = fpfh(obs); bestT = None
    for _ in range(6):
        res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            model, obs, model_fpfh, obs_fpfh, True, VOX*1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOX*1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999))
        icp = o3d.pipelines.registration.registration_icp(
            model, obs, VOX*1.2, res.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        if bestT is None or icp.fitness > bestT[1]:
            bestT = (icp.transformation, icp.fitness)
    T_est = np.array(bestT[0])
    t_err = np.linalg.norm(T_est[:3,3] - gt_pose[:3,3]) * 1000
    dR = T_est[:3,:3].T @ gt_pose[:3,:3]
    r_err = np.degrees(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1)))
    est_xyz = T_est[:3,3].tolist()
    print(f"[C] pose estimada (depth REAL): xyz={np.round(est_xyz,3).tolist()}  "
          f"fitness={bestT[1]:.2f}  t_err={t_err:.1f}mm  R_err={r_err:.1f}°")

    # ---- Fase D: PICK REAL con la pose estimada (IK + snap+attach) ----
    # Dejar el bracket en la pose asentada como config inicial (evita el reset).
    sim.setObjectPosition(obj, -1, settled_pos)
    sim.setObjectQuaternion(obj, -1, settled_quat)
    frames = REPO / "experiments/results/exp27_text_to_cad/e2e_frames"
    result = run_pick_sequence(
        bridge, frames,
        target_object="/bracket_t2c",
        pose_override_xyz=est_xyz,                    # <-- pose de percepción real
        pose_source="cad_pose_from_real_coppelia_depth",
    )
    print(f"[D] PICK: proximity={result.tip_grasp_proximity_m*100:.1f}cm  "
          f"plausible={result.grasp_plausible}  object_moved={result.obj_moved_m*100:.1f}cm  "
          f"ik_converged={result.ik_converged}")

# ---- Guardar reporte ----
report = {
    "objeto": "bracket_text2cad (escuadra en L 60x40x45mm)",
    "percepcion": {
        "fuente_depth": "camara CoppeliaSim (real, render + reescala near/far real)",
        "mask": "segmentacion por color (mask del detector)",
        "depth_shape": list(depth.shape),
        "err_centroide_cm": round(float(derr)*100, 1),
    },
    "pose": {
        "gt_xyz_m": [round(x,4) for x in gt_pose[:3,3].tolist()],
        "est_xyz_m": [round(x,4) for x in est_xyz],
        "t_err_mm": round(float(t_err), 1),
        "R_err_deg": round(float(r_err), 1),
        "fitness": round(float(bestT[1]), 3),
    },
    "pick": {
        "mecanismo": "IK + snap+attach (run_pick_sequence, real del TFM)",
        "pose_source": "cad_pose_from_real_coppelia_depth",
        "tip_grasp_proximity_cm": round(result.tip_grasp_proximity_m*100, 1),
        "grasp_plausible": bool(result.grasp_plausible),
        "object_moved_cm": round(result.obj_moved_m*100, 1),
        "ik_converged": bool(result.ik_converged),
    },
}
out = REPO / "experiments/results/exp27_text_to_cad/e2e_report.json"
out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print("\nreporte:", out)
print(json.dumps(report, indent=2, ensure_ascii=False))
