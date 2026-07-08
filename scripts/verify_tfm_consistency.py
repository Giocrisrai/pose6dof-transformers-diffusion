#!/usr/bin/env python3
"""
Verificador de consistencia del TFM COMPLETO — garantía re-ejecutable.

Cruza los números canónicos de las hipótesis (fuente de verdad = los reports
JSON de experiments/results) contra dónde se citan: Entrega 3, Entrega 4 y el
dashboard. Garantiza que NO hay contradicciones en la tesis.

Claims verificados:
  H1 (precisión): AUC ADD-S 0.908/0.957 [IC95], recall 95.8%/99.7%, Δ+3.0/+3.6pp
  H3 (viabilidad sin GPU): cycle p95 6.29 s / 6.68 s (< 10 s)

Salida: informe PASS/FAIL. Código 0 si todo coherente.
"""
import json
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "experiments/results"

# ── Fuente de verdad: reports JSON ──────────────────────────────────
metrics = json.loads((RES / "local_metrics_with_bootstrap.json").read_text())
e2e = json.loads((RES / "pipeline_e2e/e2e_live_metrics.json").read_text())
yc = metrics["datasets"]["ycbv"]
tl = metrics["datasets"]["tless"]
lb = metrics["reference_leaderboard"]

CANON = {
    "auc_ycbv": round(yc["auc_adds_50mm"], 3),          # 0.908
    "auc_tless": round(tl["auc_adds_50mm"], 3),         # 0.957
    "rec_ycbv": round(yc["recall_adds_10mm"] * 100, 1),  # 95.8
    "rec_tless": round(tl["recall_adds_10mm"] * 100, 1),  # 99.7
    "dpp_ycbv": round(lb["delta_mean_ar_pp"]["ycbv"], 1),  # 3.0
    "dpp_tless": round(lb["delta_mean_ar_pp"]["tless"], 1),  # 3.6
    "p95_ycbv": round(e2e["datasets"]["ycbv"]["total_ms"]["p95"] / 1000, 2),  # 6.29
    "p95_tless": round(e2e["datasets"]["tless"]["total_ms"]["p95"] / 1000, 2),  # 6.68
    "ci_ycbv_lo": round(yc["auc_adds_50mm_ci95"]["lo"], 3),  # 0.901
    "ci_ycbv_hi": round(yc["auc_adds_50mm_ci95"]["hi"], 3),  # 0.916
}

checks: list[tuple[str, bool, str]] = []
def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))

def num_in(text, value, decimals):
    """¿Aparece `value` (con . o , decimal) en text?"""
    dot = f"{value:.{decimals}f}"
    return dot in text or dot.replace(".", ",") in text

def docx_text(path):
    if not path.exists():
        return None
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8", "ignore")
    return re.sub(r"<[^>]+>", "", xml)

# ── 1. Fuente de verdad internamente coherente ──────────────────────
check("1.AUC en rango [0,1]", 0.5 < CANON["auc_ycbv"] < 1 and 0.5 < CANON["auc_tless"] < 1)
check("1.p95 < umbral H3 (10 s)", CANON["p95_ycbv"] < 10 and CANON["p95_tless"] < 10,
      f"p95={CANON['p95_ycbv']}/{CANON['p95_tless']}s")
check("1.CI95 contiene el punto", yc["auc_adds_50mm_ci95"]["lo"] <= yc["auc_adds_50mm"] <= yc["auc_adds_50mm_ci95"]["hi"])
check("1.FP supera baseline (Δ>0)", CANON["dpp_ycbv"] > 0 and CANON["dpp_tless"] > 0)

# ── 2/3. Entrega 4 y Entrega 3 citan los mismos números ─────────────
for tag, docx in [("E4", REPO / "docs/entrega4/TFM_Entrega4_UNIR.docx"),
                  ("E3", REPO / "docs/entrega3/TFM_Entrega3_UNIR.docx")]:
    t = docx_text(docx)
    if t is None:
        check(f"{tag}.docx presente", False, "no encontrado")
        continue
    check(f"{tag}.AUC 0.908/0.957", num_in(t, CANON["auc_ycbv"], 3) and num_in(t, CANON["auc_tless"], 3))
    check(f"{tag}.cycle p95 6.29/6.68 s", num_in(t, CANON["p95_ycbv"], 2) and num_in(t, CANON["p95_tless"], 2))
    check(f"{tag}.recall 95.8/99.7 %", num_in(t, CANON["rec_ycbv"], 1) and num_in(t, CANON["rec_tless"], 1))
    check(f"{tag}.H1 y H3 aceptadas", ("H1" in t and "H3" in t and "aceptada" in t.lower()))

# ── 4. Dashboard coincide ───────────────────────────────────────────
dash = (REPO / "dashboard.py").read_text()
check("4.dash AUC 0.908/0.957", num_in(dash, CANON["auc_ycbv"], 3) and num_in(dash, CANON["auc_tless"], 3))
check("4.dash cycle p95 6.29/6.68", num_in(dash, CANON["p95_ycbv"], 2) and num_in(dash, CANON["p95_tless"], 2))
check("4.dash recall 95.8/99.7", num_in(dash, CANON["rec_ycbv"], 1) and num_in(dash, CANON["rec_tless"], 1))
check("4.dash Δ +3.0/+3.6 pp", num_in(dash, CANON["dpp_ycbv"], 1) and num_in(dash, CANON["dpp_tless"], 1))
check("4.dash CI95 0.901–0.916", num_in(dash, CANON["ci_ycbv_lo"], 3) and num_in(dash, CANON["ci_ycbv_hi"], 3))

# ── Informe ─────────────────────────────────────────────────────────
print("\n=== VERIFICACIÓN DE CONSISTENCIA — TFM COMPLETO ===")
print(f"Canónicos (fuente: reports JSON): {CANON}\n")
n_ok = sum(1 for _, ok, _ in checks if ok)
for name, ok, detail in checks:
    print(f"  [{'PASS' if ok else 'FALLA'}] {name}" + (f"  ({detail})" if detail else ""))
print(f"\n{n_ok}/{len(checks)} comprobaciones OK")
sys.exit(0 if n_ok == len(checks) else 1)
