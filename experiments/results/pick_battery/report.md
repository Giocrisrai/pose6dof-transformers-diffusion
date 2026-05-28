# Pick Battery Report — métricas honestas

**IMPORTANTE — leé `docs/PICK_LIMITATIONS.md` antes de interpretar.**

El "grasp" usa la técnica de snap+attach (cubo se teletransporta al TCP y se parentea al gripper). El gripper físico NO agarra por fricción. Esto es estándar en sims comerciales (Pickit, Cognex, RoboDK) pero hay que entender las limitaciones.

## Métricas

- **moved**: desplazamiento total del cubo (cm). Ruidoso por no-determinismo de la física post-release.
- **grasp_proximity**: distancia tip↔cubo AL momento del attach (cm). Si > 5 cm el grasp NO sería físicamente plausible.
- **deposit_error**: distancia entre obj_end y el target deposit (cm). Mide la precisión del depósito (independiente de no-determinismo).
- **ik_converged**: True si todas las llamadas a IK convergieron.

## Resultados

| id | diff | frames | grasp_prox | grasp_OK | moved | deposit_err | deposit_OK | ik_ok | video |
|---|---|---:|---:|:-:|---:|---:|:-:|:-:|---|
| base | easy | 870 | 0.8cm | ✓ | 75.6cm | 5.0cm | ✓ | ✓ | `experiments/results/pick_battery/base/demo.mp4` |
| easy | easy | 870 | 0.8cm | ✓ | 76.8cm | 6.3cm | ✓ | ✓ | `experiments/results/pick_battery/easy/demo.mp4` |
| hard | hard | 870 | 0.8cm | ✓ | 75.7cm | 4.3cm | ✓ | ✓ | `experiments/results/pick_battery/hard/demo.mp4` |
