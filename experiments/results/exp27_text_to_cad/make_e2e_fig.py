"""Figura de percepción real del E2E: RGB, depth y nube segmentada + CAD registrado."""
import sys; from pathlib import Path; import numpy as np
REPO=Path("/Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm"); sys.path.insert(0,str(REPO))
import open3d as o3d, trimesh, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCR=Path(__file__).resolve().parent/"figs"
D=REPO/"experiments/results/exp27_text_to_cad"
rgb=np.load(SCR/"e2e_rgb.npy"); depth=np.load(SCR/"e2e_depth.npy")
rep=json.loads((D/"e2e_report.json").read_text())

# reconstruir nube y registro (reproducible)
T_wc=np.array([[1,0,0,0.26],[0,-1,0,0.10],[0,0,-1,1.30],[0,0,0,1]],float)
fx=fy=554.3;cx=320;cy=240
r,g,b=rgb[:,:,0].astype(int),rgb[:,:,1].astype(int),rgb[:,:,2].astype(int)
mask=(r>110)&(g<90)&(b<90)
H,W=depth.shape; us,vs=np.meshgrid(np.arange(W),np.arange(H))
sel=mask.reshape(-1)&(depth.reshape(-1)>0.06)&(depth.reshape(-1)<1.95)
d=depth.reshape(-1)[sel];u=(W-1-us.reshape(-1)[sel]);v=(H-1-vs.reshape(-1)[sel])
pc=np.stack([(u-cx)/fx*d,(v-cy)/fy*d,d],1); world=(T_wc[:3,:3]@pc.T).T+T_wc[:3,3]

VOX=0.003
def topc(p):
    q=o3d.geometry.PointCloud(); q.points=o3d.utility.Vector3dVector(p); return q
tm=trimesh.load(D/"assets/test_bracket.stl"); tm.apply_scale(0.001)
model=topc(tm.sample(6000)).voxel_down_sample(VOX)
model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*3,max_nn=30))
obs=topc(world).voxel_down_sample(VOX)
obs.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*3,max_nn=30))
ff=lambda pc:o3d.pipelines.registration.compute_fpfh_feature(pc,o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*5,max_nn=100))
mf,of=ff(model),ff(obs); best=None
for _ in range(6):
    res=o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        model,obs,mf,of,True,VOX*1.5,o3d.pipelines.registration.TransformationEstimationPointToPoint(False),3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOX*1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(200000,0.999))
    icp=o3d.pipelines.registration.registration_icp(model,obs,VOX*1.2,res.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    if best is None or icp.fitness>best[1]: best=(np.array(icp.transformation),icp.fitness)
Te=best[0]
mp=np.asarray(model.points); reg=(Te[:3,:3]@mp.T).T+Te[:3,3]
op=np.asarray(obs.points)

fig=plt.figure(figsize=(13,4.2))
ax1=fig.add_subplot(1,3,1); ax1.imshow(rgb); ax1.contour(mask,colors='lime',linewidths=0.8); ax1.set_title("RGB real + máscara objeto"); ax1.axis('off')
ax2=fig.add_subplot(1,3,2); im=ax2.imshow(depth,cmap='viridis'); ax2.set_title("Depth real (CoppeliaSim)"); ax2.axis('off'); fig.colorbar(im,ax=ax2,fraction=0.046,label='m')
ax3=fig.add_subplot(1,3,3,projection='3d')
ax3.scatter(op[:,0]*100,op[:,1]*100,op[:,2]*100,s=6,c='#c53030',label='nube depth real')
ax3.scatter(reg[:,0]*100,reg[:,1]*100,reg[:,2]*100,s=3,c='#2b6cb0',alpha=.5,label='CAD registrado')
ax3.set_title(f"Registro: t_err={rep['pose']['t_err_mm']}mm  R_err={rep['pose']['R_err_deg']}°")
ax3.set_xlabel('X(cm)');ax3.set_ylabel('Y(cm)');ax3.set_zlabel('Z(cm)');ax3.legend(fontsize=7);ax3.view_init(22,-70)
fig.suptitle("E2E real: depth de CoppeliaSim → pose 6-DoF model-based del bracket text-to-CAD",fontsize=11)
plt.tight_layout(); plt.savefig(D/"figs/e2e_perception.png",dpi=120); print("figura OK; fitness",round(best[1],2))
