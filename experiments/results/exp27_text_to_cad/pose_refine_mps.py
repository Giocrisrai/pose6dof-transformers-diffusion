"""
Refinador de pose 6-DoF por gradiente en Apple MPS (GPU del M1) — local, sin CUDA.

Análogo local del refiner de FoundationPose: parte de una hipótesis global clásica
(FPFH+RANSAC) y refina la pose SE(3) minimizando, por descenso de gradiente con
autograd de PyTorch sobre la GPU Metal (MPS), la distancia de la nube de profundidad
REAL observada al modelo CAD. Compara con la ground-truth exacta.
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh

REPO = Path("/Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm")
EXP = REPO / "experiments/results/exp27_text_to_cad"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"dispositivo: {DEV}  (torch {torch.__version__})")

# --- datos reales capturados en CoppeliaSim ---
rgb = np.load(EXP/"figs/e2e_rgb.npy"); depth = np.load(EXP/"figs/e2e_depth.npy")
rep = json.loads((EXP/"e2e_report.json").read_text())
gt = np.array(rep["pose"]["gt_xyz_m"])
K = np.array([[554.3,0,320.0],[0,554.3,240.0],[0,0,1.0]])
T_wc = np.array([[1,0,0,0.26],[0,-1,0,0.10],[0,0,-1,1.30],[0,0,0,1.0]])

# nube observada desde depth+mask reales (convención validada en exp27)
H,W = depth.shape; us,vs = np.meshgrid(np.arange(W),np.arange(H))
r,g,b = rgb[:,:,0].astype(int),rgb[:,:,1].astype(int),rgb[:,:,2].astype(int)
mask = (r>110)&(g<90)&(b<90)
sel = mask.reshape(-1)&(depth.reshape(-1)>0.06)&(depth.reshape(-1)<1.95)
d = depth.reshape(-1)[sel]; u=(W-1-us.reshape(-1)[sel]); v=(H-1-vs.reshape(-1)[sel])
pc = np.stack([(u-320)/554.3*d,(v-240)/554.3*d,d],1)
obs = (T_wc[:3,:3]@pc.T).T + T_wc[:3,3]           # nube observada en mundo (m)

# modelo CAD (con normales por punto para point-to-plane)
tm = trimesh.load(EXP/"assets/test_bracket.stl"); tm.apply_scale(0.001)
model, face_idx = trimesh.sample.sample_surface(tm, 2500)
model_n = tm.face_normals[face_idx]

# --- init: hipótesis global clásica (FPFH+RANSAC) en CPU ---
def pcd(p):
    q=o3d.geometry.PointCloud(); q.points=o3d.utility.Vector3dVector(p); return q
VOX=0.003
mo=pcd(model).voxel_down_sample(VOX); ob=pcd(obs).voxel_down_sample(VOX)
for p in (mo,ob): p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*3,max_nn=30))
ff=lambda x:o3d.pipelines.registration.compute_fpfh_feature(x,o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*5,max_nn=100))
res=o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    mo,ob,ff(mo),ff(ob),True,VOX*1.5,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),3,
    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOX*1.5)],
    o3d.pipelines.registration.RANSACConvergenceCriteria(100000,0.999))
T0=np.array(res.transformation)
print("init global (FPFH) fitness:", round(res.fitness,3))

# --- refiner por gradiente en MPS ---
def r6d_to_R(x):  # rotación 6D (Zhou et al. 2019) -> matriz 3x3 (diferenciable)
    a1,a2 = x[:3], x[3:]
    b1 = a1/ a1.norm()
    b2 = a2 - (b1*a2).sum()*b1; b2 = b2/b2.norm()
    b3 = torch.linalg.cross(b1,b2)
    return torch.stack([b1,b2,b3],dim=1)

M = torch.tensor(np.asarray(model), dtype=torch.float32, device=DEV)   # (N,3) modelo
MN = torch.tensor(np.asarray(model_n), dtype=torch.float32, device=DEV) # (N,3) normales
O = torch.tensor(np.asarray(ob.points), dtype=torch.float32, device=DEV)  # (P,3) obs

# Perturbar la hipótesis inicial (simula una hipótesis burda, como las que
# muestrea FoundationPose antes del refiner): +25° de rotación y +3 cm.
rng = np.random.default_rng(7)
ax = rng.normal(size=3); ax/=np.linalg.norm(ax); ang=np.deg2rad(25)
Kx=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
Rpert=np.eye(3)+np.sin(ang)*Kx+(1-np.cos(ang))*(Kx@Kx)
T_init = T0.copy(); T_init[:3,:3]=Rpert@T0[:3,:3]; T_init[:3,3]=T0[:3,3]+np.array([0.02,-0.02,0.015])
# init params desde la hipótesis perturbada
R0 = torch.tensor(T_init[:3,:3], dtype=torch.float32, device=DEV)
r6 = torch.cat([R0[:,0],R0[:,1]]).clone().requires_grad_(True)
t = torch.tensor(T_init[:3,3], dtype=torch.float32, device=DEV).clone().requires_grad_(True)
opt = torch.optim.Adam([r6,t], lr=0.004)

def loss_fn(R, t):
    Mt = M @ R.T + t                     # modelo transformado
    Nt = MN @ R.T                        # normales transformadas
    d2 = torch.cdist(O, Mt)              # (P,N)
    idx = d2.argmin(dim=1)               # vecino más cercano por punto observado
    mnn = Mt[idx]; nnn = Nt[idx]
    p2plane = (((O - mnn) * nnn).sum(1))**2       # distancia punto-plano (ICP fino)
    p2point = ((O - mnn)**2).sum(1)               # término punto-punto (estabilidad)
    return (p2plane.mean() + 0.1*p2point.mean())

import time

losses=[]; t0=time.time()
for it in range(500):
    opt.zero_grad()
    loss = loss_fn(r6d_to_R(r6), t)
    loss.backward(); opt.step()
    losses.append(loss.item())
dt=time.time()-t0

# pose refinada: el origen del objeto en mundo = traslación de la pose (Tf @ [0,0,0,1])
Rf = r6d_to_R(r6).detach().cpu().numpy(); tf = t.detach().cpu().numpy()
Tf = np.eye(4); Tf[:3,:3]=Rf; Tf[:3,3]=tf
def t_err_of(T): return np.linalg.norm(T[:3,3] - gt)*1000
def r_err_of(T):  # error de rotación vs la hipótesis global FPFH (referencia de orientación)
    dR=T[:3,:3].T@T0[:3,:3]; return np.degrees(np.arccos(np.clip((np.trace(dR)-1)/2,-1,1)))
err_init=t_err_of(T_init); err_ref=t_err_of(Tf)
print(f"\n=== refiner MPS ({DEV}) — recuperación desde hipótesis perturbada ===")
print(f"iteraciones: {len(losses)} en {dt:.2f}s ({len(losses)/dt:.0f} it/s) en la GPU del M1")
print(f"loss point-to-plane: {losses[0]*1e6:.1f} -> {losses[-1]*1e6:.2f} um^2")
print(f"error traslación:  hipótesis perturbada {err_init:.1f} mm  ->  refinado(MPS) {err_ref:.1f} mm")
print(f"error rotación vs FPFH:  init {r_err_of(T_init):.1f}°  ->  refinado {r_err_of(Tf):.1f}°")
print(f"pose refinada xyz: {np.round(tf,4).tolist()}   GT: {np.round(gt,4).tolist()}")
FIGS = EXP/"figs"
np.save(FIGS/"mps_refine.npy", {"losses":np.array(losses), "Tf":Tf, "T_init":T_init,
        "obs":np.asarray(ob.points), "model":np.asarray(model), "gt":gt,
        "err_init":err_init, "err_ref":err_ref, "it_per_s":len(losses)/dt, "device":DEV},
        allow_pickle=True)
print("datos ->", FIGS/"mps_refine.npy")
