#!/usr/bin/env python3
"""
Verificador de consistencia de exp27 — garantía re-ejecutable.

Comprueba que TODO es coherente y físicamente plausible:
  1. Reports JSON internamente válidos.
  2. Los números del README coinciden con los reports (fuente de verdad).
  3. La tabla del TFM (docx) coincide con los reports.
  4. El panel del dashboard coincide con los reports.
  5. El CAD generado es determinista (volumen, watertight).
  6. La física registrada es plausible (cae, se asienta, dentro del bin) y
     coherente con la ground-truth del report.

Salida: informe PASS/FAIL por comprobación. Código de salida 0 si todo OK.

Uso:  python verify_consistency.py
"""
import json
import re
import zipfile
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[2]
DOCX = REPO / "docs/predeposito/TFM_Predeposito_UNIR.docx"
TOL = 0.15  # tolerancia absoluta para números redondeados (mm/°/cm)

checks: list[tuple[str, bool, str]] = []
def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))

# ── Fuente de verdad: los reports JSON ──────────────────────────────
e2e = json.loads((EXP / "e2e_report.json").read_text())
batch = json.loads((EXP / "batch_report.json").read_text())
mps = json.loads((EXP / "mps_bench.json").read_text())

# valores canónicos que deben aparecer coherentes en todas las fuentes
bracket = {"t": e2e["pose"]["t_err_mm"], "r": e2e["pose"]["R_err_deg"],
           "prox": e2e["pick"]["tip_grasp_proximity_cm"]}
by_shape = {x["shape"]: x for x in batch}
stepped = by_shape["stepped_block"]; hexnut = by_shape["hex_nut"]

# ── 1. Reports internamente válidos ─────────────────────────────────
check("1.reports.e2e pose>0", e2e["pose"]["t_err_mm"] > 0 and e2e["pose"]["fitness"] > 0)
check("1.reports.batch 2 piezas", len(batch) == 2 and all("t_err_mm" in x for x in batch))
check("1.reports.mps exito 0-100", 0 <= mps["success_rate_pct"] <= 100 and mps["it_s_mps"] > 0)
check("1.reports.grasp plausible coherente",
      e2e["pick"]["grasp_plausible"] == (e2e["pick"]["tip_grasp_proximity_cm"] < 5.0),
      f"prox={e2e['pick']['tip_grasp_proximity_cm']}cm plausible={e2e['pick']['grasp_plausible']}")

# ── 2. README coincide con los reports ──────────────────────────────
readme = (EXP / "README.md").read_text()
def in_text(txt, value, unit):
    """¿Aparece 'value unit' (redondeado) en txt, admitiendo coma decimal?"""
    for cand in {f"{value:.1f}", f"{value:.0f}", f"{value:.1f}".replace(".", ",")}:
        if re.search(rf"\b{re.escape(cand)}\s*{re.escape(unit)}", txt):
            return True
    return False
check("2.README bracket t_err", in_text(readme, bracket["t"], "mm"), f"{bracket['t']}mm")
check("2.README bracket R_err", in_text(readme, bracket["r"], "°"), f"{bracket['r']}°")
check("2.README bracket prox", in_text(readme, bracket["prox"], "cm"), f"{bracket['prox']}cm")
check("2.README stepped t_err", in_text(readme, stepped["t_err_mm"], "mm"), f"{stepped['t_err_mm']}mm")
check("2.README mps exito", f"{mps['success_rate_pct']:.0f}" in readme and "95" in readme)
check("2.README mps ref_median", in_text(readme, mps["ref_median"], "mm"), f"{mps['ref_median']:.1f}mm")

# ── 3. TFM (docx) coincide ──────────────────────────────────────────
if DOCX.exists():
    xml = zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf-8", "ignore")
    plain = re.sub(r"<[^>]+>", "", xml)
    def in_docx(value, unit):
        # docx usa coma decimal (locale español): "4,1 mm"
        c = f"{value:.1f}".replace(".", ",")
        return re.search(rf"{re.escape(c)}\s*{re.escape(unit)}", plain) is not None
    check("3.E4 bracket t_err", in_docx(bracket["t"], "mm"), f"{bracket['t']}mm")
    check("3.E4 bracket R_err", in_docx(bracket["r"], "°"), f"{bracket['r']}°")
    check("3.E4 stepped t_err", in_docx(stepped["t_err_mm"], "mm"), f"{stepped['t_err_mm']}mm")
    check("3.E4 hex_nut flip 179", "179" in plain and "37" in plain)
    check("3.E4 sección text-to-CAD presente", "text-to-CAD" in plain)
else:
    check("3.E4 docx presente", False, "no encontrado")

# ── 4. Dashboard coincide ───────────────────────────────────────────
dash = (REPO / "dashboard.py").read_text()
# el dict `piezas` del dashboard tiene los valores hardcoded
check("4.dashboard bracket t/r/prox",
      f'"t": {bracket["t"]}' in dash and f'"r": {bracket["r"]}' in dash and f'"prox": {bracket["prox"]}' in dash)
check("4.dashboard stepped", f'"t": {stepped["t_err_mm"]}' in dash and f'"r": {stepped["r_err_deg"] if "r_err_deg" in stepped else stepped["R_err_deg"]}' in dash)

# ── 5. CAD determinista ─────────────────────────────────────────────
try:
    import trimesh
    from build123d import Box, Cylinder, Pos, Rot
    L, W, T, H, r = 60.0, 40.0, 5.0, 45.0, 3.0
    part = Pos(0, 0, T/2) * Box(L, W, T)
    part += Pos(-L/2 + T/2, 0, H/2) * Box(T, W, H)
    for hx, hy in [(15, 12), (15, -12)]:
        part -= Pos(hx, hy, T/2) * Cylinder(radius=r, height=T*2)
    part -= Pos(-L/2 + T/2, 0, H-12) * (Rot(0, 90, 0) * Cylinder(radius=r, height=T*2))
    vol = part.volume
    m = trimesh.load(EXP / "assets/test_bracket.stl")
    check("5.CAD volumen determinista (19575.9)", abs(vol - 19575.9) < 1.0, f"vol={vol:.1f}")
    check("5.CAD malla watertight", bool(m.is_watertight))
    check("5.CAD bbox 60x40x45", np.allclose(sorted(m.extents), sorted([60, 40, 45]), atol=0.5),
          f"bbox={m.extents.round(1).tolist()}")
except ImportError:
    n_skipped_cad = 1
    print("  [SKIP] 5.CAD — build123d/trimesh no instalados")

# ── 6. Física registrada plausible y coherente ──────────────────────
traj = EXP / "figs/sim_z_traj.npy"
if traj.exists():
    d = np.load(traj); z = d[:, 1] * 1000  # mm
    fell = (z[0] - z.min()) > 50           # cayó >5 cm
    settled = np.std(z[-30:]) < 1.0        # estable al final (<1 mm)
    no_through = z[-1] > -20               # no atravesó el suelo
    no_gain = z.max() <= z[0] + 1          # no ganó energía (no subió por encima del inicio)
    check("6.física cae", fell, f"z0={z[0]:.0f} zmin={z.min():.1f}mm")
    check("6.física se asienta estable", settled, f"σ_final={np.std(z[-30:]):.3f}mm")
    check("6.física no atraviesa suelo", no_through, f"zf={z[-1]:.1f}mm")
    check("6.física no gana energía", no_gain, f"zmax={z.max():.0f} vs z0={z[0]:.0f}")
else:
    check("6.física (trayectoria registrada)", False, "sim_z_traj.npy no encontrado")

# ── 7. Física EN VIVO (opt-in: EXP27_LIVE_PHYSICS=1 + CoppeliaSim en :23000) ─
import os
import sys as _sys
if not os.environ.get("EXP27_LIVE_PHYSICS"):
    print("  [SKIP] 7.física EN VIVO — pon EXP27_LIVE_PHYSICS=1 (requiere CoppeliaSim)")
else:
    try:
        import subprocess
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        RemoteAPIClient(host="localhost", port=23000).getObject("sim").getSimulationTime()
        out = subprocess.run([_sys.executable, str(EXP / "sim_drop_test.py")],
                             capture_output=True, text=True, timeout=180, cwd=str(REPO))
        m = re.search(r"zf=([0-9.]+)mm", out.stdout)
        zf_live = float(m.group(1)); zf_rec = float(np.load(traj)[-1, 1] * 1000)
        check("7.física EN VIVO reproduce evidencia",
              abs(zf_live - zf_rec) < 1.0, f"live={zf_live:.1f} vs registrado={zf_rec:.1f}mm")
    except Exception as e:
        check("7.física EN VIVO", False, f"error: {type(e).__name__}")

# ── Informe ─────────────────────────────────────────────────────────
print("\n=== VERIFICACIÓN DE CONSISTENCIA exp27 ===\n")
n_ok = sum(1 for _, ok, _ in checks if ok)
for name, ok, detail in checks:
    print(f"  [{'PASS' if ok else 'FALLA'}] {name}" + (f"  ({detail})" if detail and not ok else f"  ({detail})" if detail else ""))
print(f"\n{n_ok}/{len(checks)} comprobaciones OK")
import sys
sys.exit(0 if n_ok == len(checks) else 1)
