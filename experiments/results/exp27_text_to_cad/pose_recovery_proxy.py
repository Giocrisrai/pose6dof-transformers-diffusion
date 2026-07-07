#!/usr/bin/env python3
"""
exp27 · text-to-CAD — Paso 2: recuperación de pose 6-DoF model-based.

FoundationPose (red neuronal) requiere CUDA y se ejecuta en Google Colab; en el
M1 Pro no corre localmente (ver src/perception/foundation_pose.py). Este script
NO ejecuta la red: es un PROXY LOCAL HONESTO del mismo principio model-based
(modelo CAD -> hipótesis global FPFH+RANSAC -> refinamiento ICP point-to-plane
-> selección por fitness, el rol del scorer), que valida que el mesh generado es
apto como modelo 6-DoF y cuantifica el error frente a la ground-truth EXACTA.

Confirma además que el mesh carga por el wrapper real del repo
(FoundationPoseEstimator.load_cad_model).

Uso:
    python pose_recovery_proxy.py
"""
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

HERE = Path(__file__).resolve().parent
STL = HERE / "assets" / "test_bracket.stl"
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
VOX = 0.002  # 2 mm

# --- Confirmar compatibilidad con el wrapper real del repo ---
try:
    import sys
    sys.path.insert(0, str(HERE.parents[2]))  # raíz del repo
    from src.perception.foundation_pose import FoundationPoseEstimator
    est = FoundationPoseEstimator(device="cpu")
    est.load_cad_model(str(STL), scale=0.001)
    print(f"[wrapper] mesh cargado en FoundationPoseEstimator: "
          f"{len(est._cad_model.vertices)} vértices\n")
except Exception as e:  # pragma: no cover
    print(f"[wrapper] aviso: {e}\n")

tm = trimesh.load(STL); tm.apply_scale(0.001)

def to_pcd(p):
    q = o3d.geometry.PointCloud(); q.points = o3d.utility.Vector3dVector(p); return q

model = to_pcd(tm.sample(6000)).voxel_down_sample(VOX)
model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*3, max_nn=30))
fpfh = lambda pc: o3d.pipelines.registration.compute_fpfh_feature(
    pc, o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*5, max_nn=100))
model_fpfh = fpfh(model)

def rand_pose(rng):
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    ang = np.deg2rad(rng.uniform(10, 45))
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R = np.eye(3) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=rng.uniform(-0.05,0.05,3)+[0,0,0.4]; return T

def synth_observation(T, rng):
    pts = (T[:3,:3] @ np.asarray(model.points).T).T + T[:3,3]
    scene = to_pcd(pts); c = T[:3,3]
    _, idx = scene.hidden_point_removal([c[0]+0.18, c[1]+0.12, -0.15],
                                        np.linalg.norm(pts.max(0)-pts.min(0))*100)
    v = np.asarray(scene.select_by_index(idx).points) + rng.normal(0, 0.001, (len(idx),3))
    obs = to_pcd(v)
    obs.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*3, max_nn=30))
    return obs

def register(obs, n_hyp=5):
    obs_fpfh = fpfh(obs); best = None
    for _ in range(n_hyp):
        res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            model, obs, model_fpfh, obs_fpfh, True, VOX*1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOX*1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999))
        icp = o3d.pipelines.registration.registration_icp(
            model, obs, VOX*1.2, res.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        s = (icp.fitness, -icp.inlier_rmse)
        if best is None or s > best[0]:
            best = (s, icp.transformation, icp.fitness)
    return best[1], best[2]

def errors(Te, Tg):
    t = np.linalg.norm(Te[:3,3]-Tg[:3,3])*1000
    a = np.degrees(np.arccos(np.clip((np.trace(Te[:3,:3].T@Tg[:3,:3])-1)/2, -1, 1)))
    return t, a

N = 12; rows = []
for s in range(N):
    rng = np.random.default_rng(1000+s)
    Tg = rand_pose(rng); obs = synth_observation(Tg, rng)
    Te, fit = register(obs); te, ae = errors(Te, Tg)
    rows.append((te, ae, fit, len(obs.points)))
    print(f"pose {s+1:2d}: t_err={te:5.1f} mm  R_err={ae:6.2f}°  fitness={fit:.2f}")
rows = np.array(rows)
print(f"\nTraslación: {rows[:,0].mean():.1f} ± {rows[:,0].std():.1f} mm "
      f"(mediana {np.median(rows[:,0]):.1f})")
print(f"Rotación  : mediana {np.median(rows[:,1]):.1f}° | "
      f"correctas(<8°) {(rows[:,1]<8).sum()}/{N} | flips(>90°) {(rows[:,1]>90).sum()}")
np.save(FIGS / "pose_recovery_results.npy", rows)
