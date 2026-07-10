#!/usr/bin/env python3
"""Smoke test end-to-end de la aplicación desplegada (dashboard + API).

Valida —sin depender de un navegador— todo lo que un cliente cargaría al abrir
la app: el HTML del dashboard, cada asset estático que referencia (vídeo demo +
poster), el streaming por rango del vídeo (HTTP 206) y los endpoints de la API
(GET + POST con datos reales del pipeline).

Sirve para verificar cualquier despliegue apuntándolo a su URL base:

    python scripts/smoke_e2e_served.py                      # http://localhost:8000
    python scripts/smoke_e2e_served.py https://tfm.onrender.com

Sale con código 0 si todo pasa, 1 si algo falla (apto para CI/monitorización).
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request as u

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")

_ok = True


def _req(path: str, method: str = "GET", body=None, headers=None):
    h = dict(headers or {})
    data = json.dumps(body).encode() if body is not None else None
    if data:
        h["Content-Type"] = "application/json"
    req = u.Request(BASE + path, data=data, method=method, headers=h)
    try:
        resp = u.urlopen(req, timeout=60)  # noqa: S310 (URL controlada por el operador)
        return resp.status, resp.headers, resp.read()
    except u.HTTPError as e:
        return e.code, e.headers, e.read()


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ok
    _ok = _ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}  {detail}")


def main() -> int:
    print(f"Smoke E2E contra {BASE}")

    print("== 1. dashboard HTML en / ==")
    st, hd, bd = _req("/")
    html = bd.decode("utf-8", "replace")
    check("GET / = 200 text/html", st == 200 and "text/html" in hd.get("content-type", ""),
          f"{len(bd)} bytes")
    check("HTML ligero (<40 KB, vídeo no embebido)", len(bd) < 40_000, f"{len(bd) // 1024} KB")
    check("sin data:video inline", "data:video" not in html)

    print("== 2. assets estáticos referenciados resuelven ==")
    refs = sorted(set(re.findall(r'(?:src|href|poster)="(assets/[^"]+)"', html)))
    check("HTML referencia >=2 assets", len(refs) >= 2, str(refs))
    for ref in refs:
        st, hd, bd = _req("/" + ref)
        check(f"/{ref}", st == 200, f"{st} {hd.get('content-type')} {len(bd)}B")

    print("== 3. streaming del vídeo (HTTP Range -> 206) ==")
    st, hd, bd = _req("/assets/demo_cine.mp4", headers={"Range": "bytes=0-2047"})
    check("Range = 206 Partial Content", st == 206,
          f"{st} content-range={hd.get('content-range')}")
    check("cabecera mp4 válida (ftyp)", b"ftyp" in bd[:64], repr(bd[4:12]))

    print("== 4. endpoints de la API ==")
    for ep in ("/api", "/health", "/models", "/metrics", "/openapi.json"):
        st, hd, bd = _req(ep)
        check(f"GET {ep} = 200 json",
              st == 200 and "application/json" in hd.get("content-type", ""))
    st, _, _ = _req("/docs")
    check("GET /docs = 200 (Swagger UI)", st == 200)

    print("== 5. POST del pipeline (datos reales) ==")
    st, _, bd = _req("/plan-grasp", "POST",
                     {"object_position": [-0.05, -0.22, 0.05], "model": "ultra",
                      "n_diffusion_steps": 25, "n_samples": 1})
    d = json.loads(bd)
    check("POST /plan-grasp 200 · trayectoria 7-DoF",
          st == 200 and d["action_dim"] == 7 and len(d["trajectory"][0]) == d["horizon"],
          f"horizon={d['horizon']} lat={d['latency_ms']:.0f}ms")
    st, _, bd = _req("/e2e", "POST",
                     {"dataset": "ycbv", "n_instances": 1, "use_ultra_model": True})
    d = json.loads(bd)
    check("POST /e2e 200 · H3 cumplida",
          st == 200 and d["h3_passed"] is True,
          f"cycle_p95={d['cycle_p95_ms']:.0f}ms h3={d['h3_passed']}")

    print("\n" + ("OK  END-TO-END: todo lo que el navegador cargaría resuelve"
                  if _ok else "FALLO  hay comprobaciones en rojo"))
    return 0 if _ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
