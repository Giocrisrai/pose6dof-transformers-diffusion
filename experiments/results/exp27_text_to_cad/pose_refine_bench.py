"""
Benchmark de optimización del refiner de pose 6-DoF en Apple MPS.

Corre el refiner render-and-compare (point-to-plane, autograd) sobre N hipótesis
perturbadas aleatoriamente y reporta la estadística de convergencia y el speedup
MPS (GPU del M1) frente a CPU. Todo local, sin CUDA.
"""
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh

EXP = Path(__file__).resolve().parent
FIGS = EXP/"figs"

# --- datos reales (misma nube de profundidad del E2E) ---
rgb = np.load(FIGS/"e2e_rgb.npy"); depth = np.load(FIGS/"e2e_depth.npy")
gt = np.array(json.loads((EXP/"e2e_report.json").read_text())["pose"]["gt_xyz_m"])
T_wc = np.array([[1,0,0,0.26],[0,-1,0,0.10],[0,0,-1,1.30],[0,0,0,1.0]])
H,W = depth.shape; us,vs = np.meshgrid(np.arange(W),np.arange(H))
r,g,b = rgb[:,:,0].astype(int),rgb[:,:,1].astype(int),rgb[:,:,2].astype(int)
mask = (r>110)&(g<90)&(b<90)
sel = mask.reshape(-1)&(depth.reshape(-1)>0.06)&(depth.reshape(-1)<1.95)
d = depth.reshape(-1)[sel]; u=(W-1-us.reshape(-1)[sel]); v=(H-1-vs.reshape(-1)[sel])
pc = np.stack([(u-320)/554.3*d,(v-240)/554.3*d,d],1)
obs = (T_wc[:3,:3]@pc.T).T + T_wc[:3,3]

tm = trimesh.load(EXP/"assets/test_bracket.stl"); tm.apply_scale(0.001)
model, fidx = trimesh.sample.sample_surface(tm, 2500); model_n = tm.face_normals[fidx]

# hipótesis global de referencia (FPFH), una vez
VOX=0.003
def pcd(p):
    q=o3d.geometry.PointCloud(); q.points=o3d.utility.Vector3dVector(p); return q
mo=pcd(model).voxel_down_sample(VOX); ob_=pcd(obs).voxel_down_sample(VOX)
for p in (mo,ob_): p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*3,max_nn=30))
ff=lambda x:o3d.pipelines.registration.compute_fpfh_feature(x,o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*5,max_nn=100))
res=o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    mo,ob_,ff(mo),ff(ob_),True,VOX*1.5,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),3,
    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOX*1.5)],
    o3d.pipelines.registration.RANSACConvergenceCriteria(100000,0.999))
T0=np.array(res.transformation)
OBS = np.asarray(ob_.points)

def r6d_to_R(x):
    a1,a2=x[:3],x[3:]; b1=a1/a1.norm(); b2=a2-(b1*a2).sum()*b1; b2=b2/b2.norm()
    return torch.stack([b1,b2,torch.linalg.cross(b1,b2)],dim=1)

def refine(T_init, device, iters=500):
    M=torch.tensor(model,dtype=torch.float32,device=device)
    MN=torch.tensor(model_n,dtype=torch.float32,device=device)
    O=torch.tensor(OBS,dtype=torch.float32,device=device)
    R0=torch.tensor(T_init[:3,:3],dtype=torch.float32,device=device)
    r6=torch.cat([R0[:,0],R0[:,1]]).clone().requires_grad_(True)
    t=torch.tensor(T_init[:3,3],dtype=torch.float32,device=device).clone().requires_grad_(True)
    opt=torch.optim.Adam([r6,t],lr=0.004)
    t0=time.time()
    for _ in range(iters):
        opt.zero_grad()
        R=r6d_to_R(r6); Mt=M@R.T+t; Nt=MN@R.T
        idx=torch.cdist(O,Mt).argmin(1); mnn=Mt[idx]; nnn=Nt[idx]
        loss=(((O-mnn)*nnn).sum(1)**2).mean()+0.1*((O-mnn)**2).sum(1).mean()
        loss.backward(); opt.step()
    if device=="mps": torch.mps.synchronize()
    dt=time.time()-t0
    tf=t.detach().cpu().numpy()
    return np.linalg.norm(tf-gt)*1000, iters/dt

def perturb(seed):
    rng=np.random.default_rng(seed)
    ax=rng.normal(size=3); ax/=np.linalg.norm(ax); ang=np.deg2rad(rng.uniform(10,30))
    Kx=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    Rp=np.eye(3)+np.sin(ang)*Kx+(1-np.cos(ang))*(Kx@Kx)
    Ti=T0.copy(); Ti[:3,:3]=Rp@T0[:3,:3]; Ti[:3,3]=T0[:3,3]+rng.uniform(-0.03,0.03,3)
    return Ti, np.linalg.norm(Ti[:3,3]-gt)*1000

DEV="mps" if torch.backends.mps.is_available() else "cpu"
N=20
init_errs=[]; ref_errs=[]; speeds=[]
for s in range(N):
    Ti,ie=perturb(s)
    re_,sp=refine(Ti,DEV)
    init_errs.append(ie); ref_errs.append(re_); speeds.append(sp)
    print(f"seed {s:2d}: init {ie:5.1f} mm -> refinado {re_:5.1f} mm  ({sp:.0f} it/s)")

# speedup MPS vs CPU (5 corridas cada uno)
def bench(dev,k=5):
    Ti,_=perturb(0); sp=[]
    for _ in range(k): sp.append(refine(Ti,dev)[1])
    return np.median(sp)
sp_mps=bench("mps") if DEV=="mps" else None
sp_cpu=bench("cpu")

init_errs=np.array(init_errs); ref_errs=np.array(ref_errs)
succ=np.mean(ref_errs<10)*100
print("\n=== resumen benchmark refiner MPS ===")
print(f"N={N} | init medio {init_errs.mean():.1f} mm -> refinado mediana {np.median(ref_errs):.1f} mm (media {ref_errs.mean():.1f})")
print(f"tasa de éxito (refinado <10 mm): {succ:.0f}%")
print(f"velocidad: MPS {sp_mps:.0f} it/s | CPU {sp_cpu:.0f} it/s | speedup x{(sp_mps/sp_cpu):.1f}" if sp_mps else f"CPU {sp_cpu:.0f} it/s")

out={"N":N,"device":DEV,"init_err_mm":init_errs.tolist(),"ref_err_mm":ref_errs.tolist(),
     "init_mean":float(init_errs.mean()),"ref_median":float(np.median(ref_errs)),
     "ref_mean":float(ref_errs.mean()),"success_rate_pct":float(succ),
     "it_s_mps":float(sp_mps) if sp_mps else None,"it_s_cpu":float(sp_cpu),
     "speedup":float(sp_mps/sp_cpu) if sp_mps else None}
(EXP/"mps_bench.json").write_text(json.dumps(out,indent=2))
np.save(FIGS/"mps_bench.npy",{"init":init_errs,"ref":ref_errs,"sp_mps":sp_mps,"sp_cpu":sp_cpu},allow_pickle=True)
print("guardado mps_bench.json")
