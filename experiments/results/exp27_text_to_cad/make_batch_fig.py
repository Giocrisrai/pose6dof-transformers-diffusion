"""Catálogo de piezas + figura consolidada de métricas del mini-batch."""
from pathlib import Path
import numpy as np, json, trimesh
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
D = Path("/Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm/experiments/results/exp27_text_to_cad")

# --- catálogo: render de las 3 piezas ---
names = [("test_bracket","escuadra en L\n60x40x45"), ("hex_nut","tuerca hexagonal\n44x38x18 (simétrica)"),
         ("stepped_block","bloque escalonado\n70x45x32")]
fig = plt.figure(figsize=(12,4))
for i,(nm,ttl) in enumerate(names):
    m = trimesh.load(D/f"assets/{nm}.stl")
    ax = fig.add_subplot(1,3,i+1,projection='3d')
    tris=m.vertices[m.faces]; n=m.face_normals; L=np.array([0.3,0.4,0.85]); L/=np.linalg.norm(L)
    sh=np.clip(n@L,0.25,1); col=np.column_stack([0.30+0.5*sh,0.45+0.45*sh,0.75+0.25*sh,np.ones_like(sh)])
    ax.add_collection3d(Poly3DCollection(tris,facecolors=col,edgecolors=(0.15,0.2,0.3,0.2),linewidths=0.1))
    e=m.extents; c=m.centroid; rng=e.max()*0.6
    ax.set_xlim(c[0]-rng,c[0]+rng);ax.set_ylim(c[1]-rng,c[1]+rng);ax.set_zlim(c[2]-rng,c[2]+rng)
    ax.set_box_aspect((1,1,1)); ax.view_init(25,-60); ax.set_title(ttl,fontsize=9); ax.tick_params(labelsize=6)
fig.suptitle("Catálogo text-to-CAD: 3 piezas generadas desde texto (geometría ground-truth exacta)",fontsize=11)
plt.tight_layout(); plt.savefig(D/"figs/catalog_shapes.png",dpi=120); plt.close()

# --- métricas consolidadas ---
batch = json.loads((D/"batch_report.json").read_text())
brk = json.loads((D/"e2e_report.json").read_text())
rows = [("bracket", brk["pose"]["t_err_mm"], brk["pose"]["R_err_deg"],
         brk["pick"]["tip_grasp_proximity_cm"], brk["pick"]["grasp_plausible"])]
for r in batch:
    rows.append((r["shape"].replace("_","\n"), r["t_err_mm"], r["R_err_deg"], r["proximity_cm"], r["grasp_plausible"]))
labels=[r[0] for r in rows]; te=[r[1] for r in rows]; re_=[r[2] for r in rows]; px=[r[3] for r in rows]
x=np.arange(len(rows))
fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(13,3.8))
a1.bar(x,te,color='#2b6cb0'); a1.set_xticks(x); a1.set_xticklabels(labels,fontsize=8)
a1.set_ylabel("mm"); a1.set_title("Error de traslación"); a1.grid(alpha=.3,axis='y')
cols=['#2b6cb0' if r<90 else '#c53030' for r in re_]
a2.bar(x,re_,color=cols); a2.set_xticks(x); a2.set_xticklabels(labels,fontsize=8)
a2.axhline(90,ls=':',color='#718096',lw=1); a2.set_ylabel("°"); a2.set_title("Error de rotación\n(rojo = flip por simetría)"); a2.grid(alpha=.3,axis='y')
cols2=['#2f855a' if r[4] else '#c53030' for r in rows]
a3.bar(x,px,color=cols2); a3.axhline(5,ls='--',color='#718096',lw=1,label='umbral 5cm')
a3.set_xticks(x); a3.set_xticklabels(labels,fontsize=8); a3.set_ylabel("cm")
a3.set_title("Proximidad grasp\n(verde = plausible)"); a3.legend(fontsize=7); a3.grid(alpha=.3,axis='y')
fig.suptitle("E2E real por pieza: pose desde depth real + grasp plausible (zona diestra UR5e)",fontsize=11)
plt.tight_layout(); plt.savefig(D/"figs/batch_metrics.png",dpi=120); plt.close()
print("figuras: catalog_shapes.png, batch_metrics.png")
for r in rows: print(f"  {r[0].replace(chr(10),' '):16s} t={r[1]}mm R={r[2]}° prox={r[3]}cm plausible={r[4]}")
