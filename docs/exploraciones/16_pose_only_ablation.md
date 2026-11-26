# Exploración 16 — ¿Aporta la imagen a la política de agarre? Ablación solo pose

**Estado**: ✅ **Cerrada** — la política condicionada solo con la pose agarra igual o mejor que la visual y no se degrada cuando cambia la escena.

**Fecha**: noviembre 2026

**Scripts**: `experiments/exp30_pose_only_ablation.py` · `experiments/eval_diffusion_iter9_sim.py --politicas v9_pose_only`
**Resultados**: `experiments/results/exp30_pose_only/exp30_results.json` · `experiments/results/pick_with_diffusion/eval_v9_pose_only.json` · `experiments/results/pick_with_diffusion/eval_v9_escena_reguardada.json`

---

## La pregunta

La política v9 recibe dos cosas: la pose de la pieza (12 valores, que en el pipeline completo vienen de la estimación 6-DoF) y un embedding de la imagen RGB-D (52 valores). La exploración 15 mostró que ese embedding la vuelve frágil: con el brazo en otra postura, o con un objeto nuevo en la imagen, apunta peor. ¿Aporta algo a cambio?

## El experimento

Misma red (`hidden_dim` 256), mismos 900 pares (pose, trayectoria) del dataset de v9, sin imagen: la condición lleva solo la pose. Entrenada desde cero, 300 épocas, unos 5 minutos en MPS. Los datos no incluyen imágenes y pesan menos de 1 MB, así que están en el repositorio (`data/pose_only_v9/`) junto con los pesos: `python experiments/exp30_pose_only_ablation.py --train` lo reproduce.

## Resultados

**Validación, sin simulador** (100 casos, mejor de 8 trayectorias, semilla 2026):

| Política | Error de agarre | Agarres bajo 5 cm |
|---|---:|---:|
| Visual (v9), con las imágenes guardadas del entrenamiento | 2.0 cm | 98 % |
| Solo pose | 1.8 cm | 100 % |

La cifra de la visual se midió con el dataset completo de v9 (700 MB, no versionado).

**Simulador** (25 agarres por condición, mismas semillas, con bandeja y soltado suave):

| Condición | Visual: agarre / pick-and-place / error | Solo pose: agarre / pick-and-place / error |
|---|---:|---:|
| Escena del entrenamiento, sin distractores | 100 % / 92 % / 2.0 cm | 100 % / 92 % / 1.4 cm |
| Escena del entrenamiento, con 2 distractores | 100 % / 88 % / 2.0 cm | 100 % / 92 % / 1.4 cm |
| Escena re-guardada (brazo en otra postura), sin distractores | 72 % / 60 % / 4.5 cm | 100 % / 92 % / 1.2 cm |

Con distractores, la de solo pose desplaza alguna pieza vecina en el 28 % de los agarres, frente al 36 % de la visual.

## Lectura

- **La imagen no aporta precisión en esta tarea.** Las demostraciones son una función determinista de la pose, así que la pose basta. Lo que se llamó "invariancia visual" en Iter 8 y 9 fue enseñar a la red a ignorar la apariencia de la pieza; el resto de la imagen siguió influyendo.
- **La imagen es la fuente de la fragilidad.** Sin ella, cambiar la postura del brazo no cambia nada.
- **Para el robot real conviene partir de la política de solo pose.** La percepción (estimación 6-DoF) entrega la pose; la política no necesita volver a mirar.
- **Dónde sí haría falta la imagen:** tareas en las que la trayectoria dependa de algo que la pose no dice (obstáculos, oclusiones, agarres alternativos). Hoy la evitación de piezas vecinas la hace la verificación previa (verify-then-act), no la imagen.

## Límites

Una sola semilla de entrenamiento y 25 agarres por condición: es evidencia clara de la dirección, no una estimación fina de la diferencia. La variante de solo pose no se ha destilado a un paso; usa 100 pasos de difusión y mejor de 8, igual que v9.
