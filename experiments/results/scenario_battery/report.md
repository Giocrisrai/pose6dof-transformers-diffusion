# Scenario Battery Report

**fp_ms** = tiempo nominal de FoundationPose (no se re-ejecuta sin GPU dedicada).
**grasp_success**: bool basado en `is_grasping()` tras estabilización. Si `gripper_present=false`, el campo se interpreta como 'no aplica'.

| id | difficulty | cycle_ms | diff_ms | sim_ms | gripper | grasp_ok | snapshot |
|---|---|---:|---:|---:|:-:|:-:|---|
| base | easy | 6032 | 1363 | 515 | — | n/a | `experiments/results/scenario_battery/snapshots/base.png` |
| easy | easy | 5790 | 1045 | 591 | — | n/a | `experiments/results/scenario_battery/snapshots/easy.png` |
| hard | hard | 5343 | 678 | 511 | — | n/a | `experiments/results/scenario_battery/snapshots/hard.png` |
