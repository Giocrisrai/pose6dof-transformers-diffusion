"""Figura del refiner de pose en Apple MPS: curva de pérdida + recuperación 3D."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FIGS = Path(__file__).resolve().parent / "figs"
D = np.load(FIGS/"mps_refine.npy", allow_pickle=True).item()
losses = D["losses"]*1e6; obs=D["obs"]; model=D["model"]; Tf=D["Tf"]; Ti=D["T_init"]

def tf(T,P): return (T[:3,:3]@P.T).T + T[:3,3]
mi = tf(Ti, model); mf = tf(Tf, model)

fig = plt.figure(figsize=(11,4.3))
a1 = fig.add_subplot(1,2,1)
a1.plot(losses, color="#2b6cb0", lw=1.6); a1.set_yscale("log")
a1.set_xlabel("iteración (Adam, autograd)"); a1.set_ylabel("pérdida point-to-plane (µm²)")
a1.set_title(f"Convergencia en {D['device'].upper()} · {D['it_per_s']:.0f} it/s")
a1.grid(alpha=.3, which="both")
a2 = fig.add_subplot(1,2,2, projection="3d")
s=100
a2.scatter(obs[:,0]*s,obs[:,1]*s,obs[:,2]*s, c="#c53030", s=8, label="nube depth real")
a2.scatter(mi[::3,0]*s,mi[::3,1]*s,mi[::3,2]*s, c="#a0aec0", s=3, alpha=.5, label=f"init perturbado ({D['err_init']:.0f} mm)")
a2.scatter(mf[::3,0]*s,mf[::3,1]*s,mf[::3,2]*s, c="#2f855a", s=3, alpha=.6, label=f"refinado MPS ({D['err_ref']:.1f} mm)")
a2.set_xlabel("X(cm)"); a2.set_ylabel("Y(cm)"); a2.set_zlabel("Z(cm)")
a2.set_title("Recuperación de la pose"); a2.legend(fontsize=7, loc="upper left"); a2.view_init(24,-68)
fig.suptitle("Refiner de pose 6-DoF por gradiente en la GPU del M1 (Apple MPS) — local, sin CUDA", fontsize=11)
plt.tight_layout(); plt.savefig(FIGS/"mps_refine.png", dpi=120)
print("figura -> figs/mps_refine.png")
