"""Figura del benchmark del refiner: init vs refinado (N semillas) + éxito."""
from pathlib import Path
import numpy as np
import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EXP = Path(__file__).resolve().parent
b = json.loads((EXP/"mps_bench.json").read_text())
ie = np.array(b["init_err_mm"]); re_ = np.array(b["ref_err_mm"])
x = np.arange(len(ie))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.scatter(x, ie, c="#a0aec0", s=30, label=f"hipótesis inicial (media {b['init_mean']:.0f} mm)")
a1.scatter(x, re_, c="#2f855a", s=30, label=f"refinado (mediana {b['ref_median']:.1f} mm)")
for i in x: a1.plot([i, i], [ie[i], re_[i]], color="#cbd5e0", lw=0.8, zorder=0)
a1.axhline(10, ls="--", color="#c53030", lw=1, label="umbral éxito 10 mm")
a1.set_xlabel("semilla de perturbación"); a1.set_ylabel("error de traslación (mm)")
a1.set_title(f"Recuperación por gradiente (N={b['N']}) · éxito {b['success_rate_pct']:.0f}%")
a1.legend(fontsize=8); a1.grid(alpha=.3)

# histograma del error refinado
a2.hist(re_, bins=np.arange(0, 14, 1.5), color="#2f855a", alpha=.8, edgecolor="white")
a2.axvline(np.median(re_), ls="--", color="#2b6cb0", lw=1.5, label=f"mediana {np.median(re_):.1f} mm")
a2.set_xlabel("error refinado (mm)"); a2.set_ylabel("nº de casos")
sp = (f"MPS {b['it_s_mps']:.0f} it/s · CPU {b['it_s_cpu']:.0f} it/s"
      if b.get("it_s_mps") else f"CPU {b['it_s_cpu']:.0f} it/s")
a2.set_title(f"Distribución del error refinado\n({sp})")
a2.legend(fontsize=8); a2.grid(alpha=.3, axis="y")
fig.suptitle("Benchmark del refiner de pose por gradiente en Apple MPS (local, sin CUDA)", fontsize=11)
plt.tight_layout(); plt.savefig(EXP/"figs/mps_bench.png", dpi=120)
print("figura -> figs/mps_bench.png")
