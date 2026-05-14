# Exploración 10 — Instrucciones secuenciales multi-step

**Estado**: ✅ **ÉXITO** — **8/8 secuencias completas (100 %)**, **20/20 pasos correctos**. Mergeado a `main`.

**Fecha**: mayo 2026

---

## Hipótesis

> Frases como *"first the red cube, then the blue sphere"* se pueden parsear
> con regex simple en sub-instrucciones, y cada sub-instrucción se ejecuta
> con el modelo multi-objeto del exp 20 sin re-entrenar nada.

## Implementación

**Parser** (regex de connectors + prefijos):
- Conectores: `, then`, `then`, `after`, `, next`, `and then`, `followed by`
- Prefijos eliminados: `first`, `in order:`, `sequence:`, `please`
- Caso especial: `in order: a, b, c` se splittea por comas si no hay connectors

**Orquestador**:
```
text → parse_sequence(text) → [sub_1, sub_2, ..., sub_k]
       for each sub_i:
           chosen, conf, trajectory = run_single_step(sub_i, remaining_objects, ...)
           remove chosen from remaining_objects   # mas realista
       return list of (sub_i, chosen, traj)
```

**Modelo reutilizado**: `diffusion_policy_clip_multi.pth` del exp 20 (N=2..5).
No requiere re-entrenamiento.

## Resultados (8 escenas curadas)

| # | Instrucción | Pasos | Resultado |
|---|---|:---:|:---:|
| 1 | "first the red cube, then the blue sphere" | 2 | ✅ 2/2 |
| 2 | "pick the green cylinder, then the red box" | 2 | ✅ 2/2 |
| 3 | "grab the blue cube and then the green sphere" | 2 | ✅ 2/2 |
| 4 | "first the green box, then the red sphere, then the blue cylinder" | 3 | ✅ 3/3 |
| 5 | "in order: red cube, blue box, green sphere" | 3 | ✅ 3/3 |
| 6 | "take the red sphere followed by the blue cube" | 2 | ✅ 2/2 |
| 7 | "first the blue sphere then the red cylinder then the green box then the red sphere" | 4 | ✅ 4/4 |
| 8 | "pick the small green cube, then the large red sphere" | 2 | ✅ 2/2 |

**Overall**: 8/8 = 100 % | **Step-level**: 20/20 = 100 %

## Casos especialmente interesantes

- **Escena 4 (3 pasos)**: 3 objetos distintos con orden no trivial.
- **Escena 5**: 4 objetos en la escena, 3 a coger (uno queda sin coger).
- **Escena 7 (4 pasos)**: secuencia más larga validada, 4 objetos completos.
- **Escena 8**: incluye modificador "small"/"large" no entrenado en el modelo
  base — CLIP lo procesa apropiadamente, no rompe el modelo.

## Lecciones aprendidas

1. **No requiere re-entrenamiento**: el modelo del exp 20 ya sabe
   "elegir 1 objeto entre N". El orquestador secuencial es una
   extensión externa.
2. **Removal después de pick** es importante: simula que el objeto deja
   de estar disponible para los siguientes pasos.
3. **Parser regex simple basta** para las plantillas típicas. Para producción
   habría que usar un LLM ligero (qwen 0.5B / phi-3 mini) que entienda más
   variantes.

## Decisión

✅ **Merge a `main`**. No se guarda modelo nuevo — reutiliza el del exp 20.

## Limitaciones

- Parser solo entiende conectores explícitos (`then`, `, then`, etc).
  Frases tipo "before grabbing the X, take the Y" requieren reorder
  semántico (LLM).
- Sin razonamiento condicional ("if there is a red one, pick it; else the blue").
- Sin chunking inteligente: la secuencia se ejecuta en orden estricto, no
  optimiza distancias.

## Implicación industrial

Patrón directamente aplicable:
- **Logística**: "first the small packages, then the large ones"
- **Reciclaje**: "pick all the PET bottles, then the cans"
- **Manufactura**: "first the bolts, then the nuts, then the washers"
- **Médico**: "dispense aspirin, then ibuprofen, then paracetamol"

## Archivos

- `experiments/exp23_sequential_instructions.py` (parser + ejecutor)
- `experiments/results/exp23_sequential/exp23_results.json`
- `experiments/results/exp23_sequential/scene_01..08.png`
- `experiments/results/exp23_sequential/grid_overview.png`
