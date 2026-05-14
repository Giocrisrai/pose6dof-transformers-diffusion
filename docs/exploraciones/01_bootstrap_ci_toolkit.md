# Exploración 1 — Bootstrap-CI BOP toolkit (PyPI)

**Estado**: ✅ **ÉXITO** — listo para merge a `main` y publicación en PyPI.

**Rama**: `explore/01-bootstrap-ci-toolkit`

**Fecha de cierre**: mayo 2026

---

## Hipótesis original

> Podemos extraer nuestro framework de evaluación con bootstrap CI 95 % B=1000
> a una librería standalone PyPI compatible con `bop_toolkit` oficial.

## Criterios de éxito y resultados

| Criterio | Objetivo | Resultado | Estado |
|---|---|---|---|
| Paquete instalable con `pip install` | ✓ | `pip install -e packages/bop_bootstrap_ci` funciona, wheel + sdist generados | ✅ |
| Funciona sobre BOP YCB-V y T-LESS sin re-escribir nada | ✓ | Acepta arrays numpy de errores per-instance directamente | ✅ |
| API compatible con outputs de `bop_toolkit` | ✓ | Las funciones consumen `np.ndarray` que es lo que produce bop_toolkit | ✅ |
| ≥ 95 % cobertura de tests | ≥ 95 % | **97 % cobertura, 27 tests pasando** | ✅ |
| Documentación con 2 ejemplos | ✓ | README + `examples/quickstart.py` | ✅ |
| Validación del recipe legacy (TFM) | bit-a-bit | `test_reproduces_tfm.py` valida bit-a-bit | ✅ |
| `twine check` PASSED | ✓ | wheel y sdist pasan validación PyPI | ✅ |

## Cómo se hizo

1. Extracción de `bootstrap_ci`, `recall_threshold` y `auc_metric` desde
   `experiments/recompute_metrics_with_bootstrap.py` (3 funciones).
2. Re-empaquetado con tipado moderno (`BootstrapResult` dataclass frozen),
   docstrings con ejemplos doctest, y `py.typed` marker para soporte mypy.
3. 21 tests unitarios + 6 tests de cross-validation contra el recipe legacy.
4. Soporte tanto numpy < 2.0 (`np.trapz`) como numpy ≥ 2.0 (`np.trapezoid`).
5. `pyproject.toml` PEP 621 con extras `[dev]`.
6. README profesional con ejemplo paper-ready y cita BibTeX al TFM.

## Métricas finales

```
27 tests passed in 1.89s
97 % branch coverage (49 stmts, 1 missed)
wheel: bop_bootstrap_ci-0.1.0-py3-none-any.whl
sdist: bop_bootstrap_ci-0.1.0.tar.gz
twine check: 2/2 PASSED
```

## Validación contra TFM (publicado)

Sobre los datos commiteados (`local_metrics_with_bootstrap.json`):

| Dataset | Métrica | TFM (oficial) | Paquete reproduce |
|---|---|---|---|
| YCB-V (n=1098) | AUC ADD-S @50 mm | 0.9084 [0.9005, 0.9157] | ✅ bit-a-bit |
| YCB-V | Recall@10 mm | 95.8 % [94.6 %, 96.9 %] | ✅ bit-a-bit |
| T-LESS (n=1012) | AUC ADD-S @50 mm | 0.9571 [0.9544, 0.9589] | ✅ bit-a-bit |
| T-LESS | Recall@10 mm | 99.7 % [99.3 %, 100.0 %] | ✅ bit-a-bit |

## Decisión

✅ **Merge a `main`**. La exploración cumple todos los criterios.

Próximos pasos (no bloqueantes):
- Publicar en TestPyPI antes que en PyPI público
- Abrir issue/PR en `thodan/bop_toolkit` proponiendo integración
- Mencionar en `docs/INNOVACION_Y_ESTADO_DEL_ARTE.md` como aporte adicional

## Limitaciones documentadas

- Solo método **percentile** (no BCa). BCa corrige sesgo pero raramente
  se reporta en BOP — mantener compatibilidad con el TFM tiene prioridad.
- Asume i.i.d. en errores per-instance. Si hay correlaciones fuertes
  por escena, considerar block bootstrap (no incluido).
- Sin corrección de comparaciones múltiples — debe aplicarse externamente.

## Archivos producidos

- `packages/bop_bootstrap_ci/pyproject.toml`
- `packages/bop_bootstrap_ci/README.md`
- `packages/bop_bootstrap_ci/src/bop_bootstrap_ci/__init__.py`
- `packages/bop_bootstrap_ci/src/bop_bootstrap_ci/ci.py`
- `packages/bop_bootstrap_ci/src/bop_bootstrap_ci/py.typed`
- `packages/bop_bootstrap_ci/tests/test_ci.py` (21 tests)
- `packages/bop_bootstrap_ci/tests/test_reproduces_tfm.py` (6 tests)
- `packages/bop_bootstrap_ci/examples/quickstart.py`
