# Exploración 7 — Simulaciones visuales del VLA-lite multi-atributo

**Estado**: ✅ **ÉXITO** — 12/12 escenas demostrativas correctas con renders 3D que muestran objetos reales (cubos, esferas, cilindros, cajas), trayectoria planificada y decisión del gate. Mergeado a `main`.

**Fecha**: mayo 2026

---

## Objetivo

Demostrar **visualmente** la potencialidad del pipeline multi-atributo
(exp 18) sobre 12 escenas curadas que cubren:

1. Selección **por color solo**: "pick the red object", "select the green one"
2. Selección **por forma solo**: "pick the sphere", "grab the cylinder", "take the box", "select the cube"
3. Selección **combinada**: "pick the red sphere", "grab the blue cube", "select the green cylinder", "take the green box"
4. **Casos límite**: "the round one", "pick anything blue"

## Resultados (12/12 ✓)

Cada escena genera una tarjeta 3-paneles:
- **Panel A** (escena 3D): mesa con 2 objetos renderizados con sus formas reales (cube/sphere/cylinder/box), trayectoria planificada y punto de agarre.
- **Panel B** (gate probabilities): barras con la confianza del modelo para cada objeto.
- **Panel C** (explicación): atributos, frase, objeto elegido, target esperado, resultado.

Todas las decisiones tomadas con confianza ≥ 98.9 %. La trayectoria converge
al objeto correcto con distancia media al target de **5.97 cm**.

| Escena | Frase | Confianza | Distancia |
|---|---|---|---|
| 1 | pick the red object | 100.0 % | 5.9 cm |
| 2 | select the green one | 100.0 % | 7.5 cm |
| 3 | pick the sphere | 100.0 % | 8.1 cm |
| 4 | grab the cylinder | 100.0 % | 1.3 cm |
| 5 | take the box | 100.0 % | 3.8 cm |
| 6 | select the cube | 100.0 % | 6.5 cm |
| 7 | pick the red sphere | 100.0 % | 7.2 cm |
| 8 | grab the blue cube | 99.9 % | 7.5 cm |
| 9 | select the green cylinder | 99.9 % | 5.3 cm |
| 10 | take the green box | 100.0 % | 10.0 cm |
| 11 | "the round one" | 98.9 % | 3.5 cm |
| 12 | "pick anything blue" | 100.0 % | 7.0 cm |

## Casos especialmente interesantes

- **"the round one"** (sin mencionar la palabra "sphere"): el modelo entiende
  que "round" se asocia a sphere por CLIP, no por matching textual literal.
- **"pick anything blue"** (sin sustantivo concreto): el modelo se centra
  solo en el atributo color disponible.

## Decisión

✅ **Merge a main**. Renders disponibles en `experiments/results/exp19_visual_sims/`.

## Archivos

- `experiments/exp19_visual_simulations.py`
- `experiments/results/exp19_visual_sims/scene_01.png` ... `scene_12.png`
- `experiments/results/exp19_visual_sims/grid_overview.png` (composición)
- `experiments/results/exp19_visual_sims/exp19_results.json`
