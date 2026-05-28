# Pick Battery Report

Cada escenario ejecuta la secuencia completa pick-and-place (home → approach → descend → grasp → lift → deposit → release → home) sobre la escena cargada con sus tweaks aplicados.

**object_manipulated**: el objeto target se desplazó más de 30 cm durante la secuencia (no garantiza grasp exitoso — solo contacto/desplazamiento).

| id | difficulty | frames | object_start → end | moved | manipulated | video |
|---|---|---:|---|---:|:-:|---|
| base | easy | 870 | (+0.46,-0.10,+0.03) → (-0.36,-0.18,+0.03) | 82.5 cm | ✓ | `experiments/results/pick_battery/base/demo.mp4` |
| easy | easy | 870 | (+0.46,-0.10,+0.03) → (-0.31,+0.61,+0.03) | 104.9 cm | ✓ | `experiments/results/pick_battery/easy/demo.mp4` |
| hard | hard | 870 | (+0.46,-0.10,+0.03) → (-0.44,-0.01,+0.03) | 90.6 cm | ✓ | `experiments/results/pick_battery/hard/demo.mp4` |
