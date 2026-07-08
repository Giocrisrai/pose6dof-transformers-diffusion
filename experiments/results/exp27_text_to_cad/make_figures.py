#!/usr/bin/env python3
"""exp27 · text-to-CAD — Regenera las figuras a partir de los .npy en figs/."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path(__file__).resolve().parent / "figs"

# --- Trayectoria z(t) de la caída en CoppeliaSim ---
d = np.load(FIGS / "sim_z_traj.npy"); t, z = d[:, 0], d[:, 1] * 1000
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, z, lw=1.8, color="#2b6cb0")
ax.axhline(z[-1], ls="--", lw=1, color="#718096", label=f"asentado (~{z[-1]:.0f} mm)")
ax.fill_between(t, 0, 6, color="#cbd5e0", alpha=.5, label="base del bin")
imin = int(np.argmin(z))
ax.annotate("impacto", xy=(t[imin], z[imin]), xytext=(t[imin] + 0.15, z[imin] + 40),
            arrowprops=dict(arrowstyle="->", color="#c53030"), color="#c53030", fontsize=9)
ax.set_xlabel("tiempo de simulación (s)"); ax.set_ylabel("altura del bracket z (mm)")
ax.set_title("CoppeliaSim: caída y asentamiento del bracket en el bin")
ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIGS / "sim_z_trajectory.png", dpi=120); plt.close()

# --- Evaluación de recuperación de pose ---
r = np.load(FIGS / "pose_recovery_results.npy"); te, ae = r[:, 0], r[:, 1]
flip = ae > 90; x = np.arange(1, len(te) + 1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.bar(x[~flip], te[~flip], color="#2b6cb0", label="pose recuperada")
a1.bar(x[flip], te[flip], color="#c53030", label="flip 180° (ambigüedad)")
a1.axhline(np.median(te), ls="--", color="#2f855a", lw=1, label=f"mediana {np.median(te):.1f} mm")
a1.set_xlabel("caso"); a1.set_ylabel("error de traslación (mm)")
a1.set_title(f"Traslación: {te.mean():.1f} ± {te.std():.1f} mm"); a1.legend(fontsize=8); a1.grid(alpha=.3, axis="y")
a2.bar(x[~flip], ae[~flip], color="#2b6cb0"); a2.bar(x[flip], ae[flip], color="#c53030")
a2.axhline(np.median(ae), ls="--", color="#2f855a", lw=1, label=f"mediana {np.median(ae):.1f}°")
a2.axhline(5, ls=":", color="#718096", lw=1, label="umbral 5°")
a2.set_xlabel("caso"); a2.set_ylabel("error de rotación (°)")
a2.set_title(f"Rotación: {(ae < 8).sum()}/{len(ae)} correctas; {flip.sum()} flips por vista parcial")
a2.legend(fontsize=8); a2.grid(alpha=.3, axis="y")
fig.suptitle("Recuperación de pose 6-DoF con el mesh text-to-CAD (proxy model-based FPFH+ICP, N=12)", fontsize=11)
plt.tight_layout(); plt.savefig(FIGS / "pose_recovery_eval.png", dpi=120); plt.close()
print("Figuras regeneradas en", FIGS)
