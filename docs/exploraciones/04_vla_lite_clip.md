# Exploración 4 — VLA-lite con CLIP text-conditioning

**Estado**: ✅ **ÉXITO** — selection accuracy **98.6 %** (criterio ≥ 75 %). Sistema entiende lenguaje natural y selecciona el objeto correcto en escenas multi-objeto. Mergeado a `main`.

**Rama**: `explore/04-vla-lite-clip`

**Fecha de cierre**: mayo 2026

---

## Hipótesis original

> Añadiendo un encoder CLIP texto al condicionamiento del Diffusion Policy,
> el sistema puede filtrar trayectorias hacia un objeto descrito en lenguaje
> natural ("pick the red object") sin requerir entrenamiento VLA completo
> (millones de demos).

## Resultados (n_train=4000, n_val=800, 40 epochs, 1.1 min en M1 Pro)

| Métrica | Objetivo | Resultado | Estado |
|---|---|---|---|
| Selection accuracy | ≥ 75 % | **98.6 %** | ✅ +23.6 pp sobre target |
| Gate accuracy (sanity) | — | 100 % | ✅ |
| Distancia media al target | mín | 11.0 cm | ✅ |
| Distancia media al distractor | máx | 46.3 cm | ✅ Δ = 35.3 cm |
| MSE vs GT trayectoria | bajo | 0.00186 | ✅ |
| Latencia DDIM-25 | < 200 ms | 1.96 ms | ✅ |
| Tests passing | sin regresión | 171/171 | ✅ |

## Lecciones aprendidas — dos intentos fallidos antes del éxito

### Intento 1: CLIP raw 32-D proyectado, sin gate (50.1 %)
**Falla**: el projector CLIP era **un módulo aleatorio inicializado pero nunca entrenado** porque las proyecciones se precomputaban una sola vez. Bug grave de pipeline.

### Intento 2: projector entrenable end-to-end, sin gate (49.8 %)
**Falla**: aunque el projector ya se entrenaba, las **32-D dispersas no daban suficiente inductive bias** para que el UNet aprendiera a discriminar entre objetos. El modelo aprendía a generar trayectorias "promedio" que terminaban en el medio entre A y B.

### Intento 3: `TextGroundedGate` + aux classification loss (98.6 %) ✅
**Solución**: añadir un **gating module explícito** que toma (CLIP_emb, RGB_a, RGB_b) y produce (gate_a, gate_b) softmax. Luego `selected_pos = gate_a · p_a + gate_b · p_b` se inyecta directamente en el cond del UNet. Adicionalmente, **auxiliary cross-entropy loss** supervisa el gate con `target_idx` (señal de cuál objeto debe seleccionarse).

**Aporte arquitectónico**: el gating module es el inductive bias clave. Sin él, el UNet no puede inferir "X color corresponde a target", aunque tenga toda la información.

## Arquitectura final

```
text "pick the red object"
       │
       ▼
   CLIP text encoder (frozen, 63 M params)
       │
       ▼
   embedding 512-D ────────┐
       │                    │
       ▼                    ▼
  CLIPProjector       TextGroundedGate
   (entrenable)       (entrenable)
       │            ┌──── gate_a ──── p_a ───┐
       │ proj 32-D  ├──── gate_b ──── p_b ───┤
       │            │                         │
       ▼            ▼                         ▼
   ┌──────────────────────────────────┐
   │  cond layout (64-D):              │
   │  [0..3]    selected_pos (gate-mix)│
   │  [4..15]   p_a, p_b, RGB_a, RGB_b │
   │  [16..18]  padding                │
   │  [19..51]  CLIP proj 32-D         │
   │  [51..63]  padding                │
   └──────────────────────────────────┘
       │
       ▼
   ConditionalUNet1D (entrenable)
       │
       ▼
   trayectoria 16 pasos hacia el objeto target
```

## Implementación

- **`CLIPProjector`** (~10 K params): MLP 512 → 64 → 32. Aprende a destilar la información de CLIP relevante para el gating.
- **`TextGroundedGate`** (~100 K params): MLP que recibe (CLIP, RGB_obj) y produce un score logit por objeto, luego softmax para gates.
- **Aux loss**: `NLLLoss(log_gates, target_idx)` con peso 0.5. Acelera el aprendizaje del gate de 50 % → 100 % en 1 epoch.
- **Total**: 1.46 M parámetros (DP + proj + gate). Entrena en 1.1 min en M1 Pro MPS.

## Datos sintéticos

- **5 templates de descripción**: "pick the {color}", "grab the {color} item", etc.
- **3 colores**: red, blue, green (no shapes para mantener simplicidad)
- **Escenas**: 2 objetos con colores DISTINTOS, separados ≥ 20 cm
- **GT trayectoria**: spline simple desde origen al target con bump Z para evitar colisiones

## Decisión

✅ **Merge a `main`**. Cumple ampliamente los criterios.

## Limitaciones honestas

1. **Vocabulario cerrado**: solo 3 colores y 5 templates. CLIP entiende muchísimo más pero no probamos generalización.
2. **Geometría sintética**: trayectorias spline simples. En el sistema real (Diffusion Policy + PBVS) las trayectorias son más complejas.
3. **No probado en escenas BOP reales**: el experimento es controlado/sintético. Para producción habría que entrenar con escenas BOP donde el "color/material" se infiere de la imagen RGB del objeto.
4. **Single-arm, no multi-step**: el modelo selecciona UN objeto entre 2. No maneja secuencias ("pick the red THEN the blue").
5. **No usa imágenes del objeto**: solo el RGB declarado del color como atributo numérico, no la apariencia visual real procesada por CLIP-image. Esto es una simplificación para mantener el experimento manejable.

## Implicaciones para el TFM

- **Sección "Trabajo futuro"**: ahora hay prueba de concepto **cuantitativa** de que el pipeline puede extenderse a VLA-lite sin necesitar millones de demos (1.1 min entrenamiento, 4000 escenas sintéticas).
- **Cierra el discurso** sobre los VLA (RDT-1B, π0): nuestro pipeline puede *acercarse* al paradigma VLA con coste muy bajo, manteniendo interpretabilidad (el gate es transparente: "el modelo eligió A con probabilidad 0.96").
- **Diferenciación**: vs RDT-1B (1.2 B params, 1 M episodios), nuestro VLA-lite es 1000× más pequeño y entrenable en portátil, a costa de un dominio mucho más restringido.

## Trabajo futuro (no bloqueante)

1. **Generalización**: añadir más colores (yellow, orange, purple) y shapes (cube, sphere, cylinder). Medir generalización a templates no vistos.
2. **Imagen real**: usar CLIP-image sobre crops del objeto para inferir atributos visuales reales (no del label).
3. **Multi-step**: extender a instrucciones secuenciales ("first the red, then the blue").
4. **Integración Gradio**: añadir un campo de texto en la UI para que usuarios prueben.

## Archivos producidos

- `experiments/exp16_vla_lite_clip.py` (script reproducible)
- `experiments/results/exp16_vla_lite/exp16_results.json`
- `data/models/diffusion_policy_clip.pth` (~5.5 MB + 63 M CLIP weights se descargan al cargar)
- `tests/test_vla_lite_modules.py` (9 tests)
