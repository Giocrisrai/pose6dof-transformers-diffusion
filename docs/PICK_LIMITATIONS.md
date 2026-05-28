# Limitaciones del Pick-and-Place Demo

> **Lectura obligatoria antes de usar `run_pick_battery.py` para defensa o venta.**

## Qué hace REALMENTE el demo

El pipeline ejecuta:
1. IK real del UR5 vía `simIK` (módulo nativo de CoppeliaSim) para mover el TCP.
2. Animación coreografiada por waypoints (home → approach → descend → ...).
3. **Snap del cubo a la pose del TCP + parent-child attach al tip dummy.**
4. Animación del lift, deposit y release.
5. Captura de frames + MP4 con ffmpeg.

## Qué NO hace

| Claim típico | Realidad |
|---|---|
| "El robot agarra el cubo con el gripper" | **NO**. El cubo se teletransporta al TCP. El cierre del gripper RG2 es cosmético. |
| "El grasp es físico (fricción dedos-cubo)" | **NO**. Sin attach el grasp falla por mala alineación finger-pad/cubo. |
| "El cubo se queda en el área de depósito" | **NO siempre**. Post-release el cubo vuela por inercia (típicamente 30-80 cm del target). |
| "Las métricas de la corrida son reproducibles" | **NO**. La física post-release es no-determinística entre corridas. |

## Por qué usamos snap+attach

La técnica de **snap+attach** (parent del objeto al end-effector durante el grasp) es la herramienta estándar usada por simuladores comerciales:

- **Pickit** (Pick-it 3D bin picking)
- **Cognex VisionPro / RoboDK**
- **NVIDIA Isaac Sim** (con `set_target_object`)
- Sims de robótica académica para demo de pipelines

**No es deshonesto** USARLA — es deshonesto **NO declarar** que la usás.

El attach permite validar:
- ✓ El pipeline percepción → planning → ejecución
- ✓ La cinemática del robot (IK)
- ✓ La coreografía de la secuencia
- ✓ Métricas de tiempo de ciclo del pipeline (no del grasp)

NO permite validar:
- ✗ La capacidad del gripper de agarrar por fricción
- ✗ La selección de pose de grasp óptima (sería trivial: cualquier pose triggera el snap)
- ✗ La robustez ante objetos resbaladizos o de geometría compleja

## Métricas honestas reportadas

`run_pick_battery.py` reporta tres métricas que **SÍ** miden algo real:

### `tip_grasp_proximity_m`
Distancia entre el TCP del UR5 y el centro del cubo en el momento ANTES del snap. Si > 5 cm, el grasp NO sería físicamente plausible (el gripper estaría lejos del cubo). **`grasp_plausible=True` significa que un gripper real podría haberlo agarrado en esa posición.**

### `deposit_error_m`
Distancia entre la posición final del cubo y el target de depósito hardcoded en (-0.30, -0.30). Mide la precisión del depósito. Como el cubo vuela post-release por inercia heredada, este número suele ser 10-80 cm. **`deposit_plausible=True` significa que terminó <30 cm del target.**

### `ik_converged`
Boolean. `True` si TODAS las llamadas a IK durante la trayectoria devolvieron `simIK.result_success`. `False` si al menos una falló (no convergió → robot puede haber quedado en pose inesperada).

## Cómo hacerlo "REAL" en una próxima iteración

| Camino | Esfuerzo | Probabilidad |
|---|---|---|
| **A:** Tunear friction coefficients del RG2 + cubos hasta lograr grasp por fricción | 4-8 h experimentación | 60% |
| **B:** Usar suction cup gripper en vez de RG2 (más permisivo a alineación) | 2-3 h | 90% |
| **C:** Constraints físicos custom en CoppeliaSim (joint constraints como Pickit lo hace) | 8-12 h con docs | 75% |
| **D:** Mantener attach + documentar honestamente (este documento) | 0 h | 100% — **lo que hicimos** |

## Para defensa de TFM

**Honesto a decir**:
- "Validamos el pipeline end-to-end con IK del UR5 nativo (simIK)"
- "El grasp usa técnica de attach (estándar en sims comerciales)"
- "Métricas: `grasp_plausible`, `deposit_error_m`, `ik_converged` — ver report"
- "El no-determinismo post-release es esperado en física rígida"

**Deshonesto a decir**:
- "El robot agarra el cubo" (sin matizar)
- "Grasp success rate = 100%" (es tautológico)
- "El cubo termina en (x,y)" (es ruidoso)
