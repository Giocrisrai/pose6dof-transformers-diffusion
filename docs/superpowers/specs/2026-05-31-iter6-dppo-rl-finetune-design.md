# Iter 6: DPPO — RL fine-tune sobre Diffusion Policy v5

## Contexto

Iter 5b cerró pick-and-place E2E al 60 %. Pero la DP solo imita el demostrador (heurístico). Iter 4 mostró el techo de imitation learning: la DP hereda los defectos del demostrador (54 % colisiones en clutter). Para superar al demostrador necesitamos un signal de reward explícito.

**DPPO** (Ren et al., NeurIPS 2024) trata la cadena de denoising de difusión como un MDP y aplica PPO sobre las predicciones de ruido. Estado del arte 2024-2025 para policy improvement sobre Diffusion Policy.

Referencia: <https://github.com/irom-lab/dppo>, <https://arxiv.org/abs/2409.00588>.

## Goal global

Demostrar que RL fine-tune mejora **`pick_and_place_success_pct` ≥ 60 % → ≥75 %** (objetivo ambicioso) y/o reduce `distractor_collision_pct` significativamente cuando se entrena en multi-object.

## Approach progresivo (3 fases)

El usuario eligió "hacer las 3 progresivamente": CoppeliaSim (validación) → PyBullet (escala) → MuJoCo MJX (full SOTA). Cada fase tiene gates claros para decidir si avanzar.

### Phase A: Proof of concept en CoppeliaSim (1-2 días)

**Goal**: validar que el loop DPPO funciona end-to-end. No esperar convergencia real.

- **Sim**: CoppeliaSim ya existente. ~10 s/episodio, sin batch.
- **Escala**: 500-1000 episodios (3-5 h training real). Suficiente para mostrar tendencia.
- **Algoritmo**: DPPO simplificado — PPO sobre los noise predictions de los últimos K timesteps de denoising.
- **Init**: DP v5 weights.
- **Eval gate**: si la curva de reward muestra **mejora monótona > +5 pp en `pick_and_place_success` rolling**, avanzar a Phase B. Si no, debugear o pivotar a algoritmo más simple.
- **Salida**: `data/models/diffusion_policy_v6_phaseA.pth`, curva training, métrica final.

### Phase B: Scaling en PyBullet (2-3 días)

**Goal**: 10k-50k episodios. Convergencia parcial. Comparable a literatura.

- **Sim**: PyBullet con UR5e URDF + cubos + cámara virtual RGB-D.
  - **Setup**: portar `bin_base.ttt` a URDF + scene XML (~1 día). Validar comportamiento físico (snap-attach, gripper RG2).
- **Escala**: 10k-50k episodios paralelos (batch de 8-16 envs en CPU multi-thread, no GPU).
- **Algoritmo**: DPPO completo (toda la cadena de denoising como MDP, action chunk MDP).
- **Eval gate**: convergencia visible en curvas reward + val_loss. **Threshold para success**: `pick_and_place_success ≥ 70 %` en PyBullet eval n=50.
- **Salida**: `data/models/diffusion_policy_v6_phaseB.pth`, comparativa CoppeliaSim eval.

### Phase C: Full SOTA en MuJoCo MJX (3-5 días + 1-2 días training)

**Goal**: 100k-500k episodios. SOTA literature-comparable.

- **Sim**: MuJoCo MJX (JAX-batched, corre en MPS M1).
  - **Setup**: portar escena a MuJoCo XML + validar (~2 días).
  - **Riesgo**: MJX MPS support es nuevo (2025); puede no batched perfectamente. Fallback: MJX CPU multi-thread.
- **Escala**: 100k-500k episodios. **Tiempo training**: 1-2 días continuos.
- **Algoritmo**: DPPO completo + entropy bonus + GAE.
- **Eval gate**: **Threshold para success**: `pick_and_place_success ≥ 80 %` en eval CoppeliaSim.
- **Salida**: `data/models/diffusion_policy_v6_phaseC.pth` (final del TFM).

## Common — para todas las fases

### Action space

7-D × 16 horizon, mismo que DP v5.

### Observation / Conditioning

- Visual: ResNet-18 encoder (frozen, `visual_encoder_iter5.pth`).
- Pose: target 12-D flat.
- Total cond 64-D, mismo que v5.

### Reward function (shaped + binary)

Per-step (durante eval del trajectory):
- `−0.1 × dist_to_cube` durante phase grasp (k < 6)
- `−0.1 × dist_to_deposit` durante phase deposit (k > 8)

Terminal (al final del trajectory):
- `+10` si `grasp_plausible AND deposit_plausible`
- `+5` si `grasp_plausible AND NOT deposit_plausible`
- `−5` si IK falla
- `−10` si `distractor_collision` (Phase B+ con multi-obj)

Reward bounded en [−20, +12] para estabilidad PPO.

### DPPO algorithm details

- Diffusion timesteps: 100 (mismo que DP v5).
- DPPO action chunk: 4 (últimos 4 timesteps de denoising).
- PPO clip ratio: 0.2.
- Value function: small MLP sobre cond (32 → 16 → 1).
- GAE λ: 0.95.
- Discount γ: 0.99.
- Entropy bonus: 0.01.
- Batch size: 64 (Phase A/B), 256 (Phase C).
- Learning rate: 3e-4.

### Eval

Todas las fases: eval final en CoppeliaSim con seed=2026, n=50. Métrica principal: `pick_and_place_success_pct`. Comparable directo con Iter 5 (60 %).

## Decisión gates progresivos

```
Phase A run
    │
    ├── reward sube > +5 pp? ─── No ─→ debug / pivotar / parar
    │
    └── Sí ─→ Phase B port + train
                │
                ├── pick_and_place_success >= 70%? ─── No ─→ documentar limitaciones, parar
                │
                └── Sí ─→ Phase C port + train
                            │
                            └── pick_and_place_success >= 80%? ─→ SOTA result para TFM
```

## Riesgos críticos

1. **DPPO no converge en este escenario**: arquitectura DP UNet pequeña (1.35 M params) + reward sparse puede dar gradientes ruidosos. Mitigación: shaped reward + warm-start con BC loss los primeros K episodios.
2. **Sim-to-sim gap**: la DP v5 fue entrenada con CoppeliaSim; eval en PyBullet/MJX puede degradar antes del fine-tune. Mitigación: eval final SIEMPRE en CoppeliaSim para comparabilidad.
3. **Reward hacking**: el agente encuentra trayectorias degeneradas (e.g., quedarse arriba del cubo sin agarrar = no se llama a IK fail = no penalty). Mitigación: terminal bonus solo si la trayectoria visita las phases correctas (grasp Y deposit ambos contactados).
4. **Tiempo de wall-clock**: 500k episodios en MJX CPU multi-thread M1 = ~3-5 días continuos. Plan: training overnight × varios días.
5. **MJX MPS support inmaduro**: puede no funcionar batched en M1. Fallback documentado: MJX CPU.

## File structure

**Nuevos**:
- `src/rl/dppo_agent.py` — algoritmo DPPO.
- `src/rl/value_net.py` — value head.
- `src/rl/replay_buffer.py` — episode buffer.
- `src/rl/reward_fn.py` — reward shaping.
- `experiments/train_dppo_coppeliasim.py` — Phase A.
- `experiments/train_dppo_pybullet.py` — Phase B.
- `experiments/train_dppo_mjx.py` — Phase C.
- `src/simulation/pybullet_bridge.py` — bridge equivalente a CoppeliaSim para PyBullet.
- `src/simulation/mjx_bridge.py` — bridge para MuJoCo MJX.
- `data/scenes/bin_base.urdf`, `data/scenes/bin_base.xml` — escenas portadas.
- `tests/test_dppo_agent.py` — unit tests.

**Modificados**:
- `experiments/run_pick_with_diffusion.py` — soporte `DP_VERSION=v6_phaseA/B/C`.
- `docs/INTEGRATION_PIPELINE.md` — sección Iter 6.
- `.gitignore` — checkpoints v6.

## Out of scope

- Multi-task (varios objetos a depositar). Iter 7+.
- Real robot deployment. Out of scope TFM.
- Adversarial robustness. Iter 8+.

## Success criteria

| Phase | Threshold mínimo | Threshold ideal |
|---|---|---|
| A (CoppeliaSim) | curva reward sube | ≥ 65 % E2E |
| B (PyBullet) | ≥ 70 % E2E | ≥ 75 % E2E |
| C (MJX) | ≥ 75 % E2E | ≥ 80 % E2E |

Si **Phase A falla**, pivotar a "PPO simple sobre DP residual" (más conservador, fallback).

## Timeline honesto

Total: **8-12 días** de trabajo focal si todo va bien. Más realista: **2-3 semanas** con debugging incluido.

Para defensa TFM: Phase A + B son suficientes para claim "DPPO mejora DP de 60 → 70-75 %". Phase C es opcional si hay tiempo.
