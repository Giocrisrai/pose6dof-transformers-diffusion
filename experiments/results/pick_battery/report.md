# Pick Battery Report

Cada escenario ejecuta la secuencia completa pick-and-place (home → approach → descend → grasp → lift → deposit → release → home) sobre la escena cargada con sus tweaks aplicados.

**object_manipulated**: el objeto target se desplazó más de 2 cm durante la secuencia (no garantiza grasp exitoso — solo contacto/desplazamiento).

| id | difficulty | frames | object_start → end | moved | manipulated | video |
|---|---|---:|---|---:|:-:|---|
| base | easy | 655 | (+0.46,-0.10,+0.29) → (+0.46,-0.10,+0.29) | 0.2 cm | ✗ | `experiments/results/pick_battery/base/demo.mp4` |
| easy | easy | 655 | (+0.46,-0.10,+0.29) → (+0.46,-0.10,+0.29) | 0.2 cm | ✗ | `experiments/results/pick_battery/easy/demo.mp4` |
| hard | hard | 655 | (+0.46,-0.10,+0.29) → (+0.46,-0.10,+0.29) | 0.2 cm | ✗ | `experiments/results/pick_battery/hard/demo.mp4` |
