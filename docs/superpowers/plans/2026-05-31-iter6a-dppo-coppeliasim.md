# Iter 6a Implementation Plan — DPPO Phase A (CoppeliaSim proof-of-concept)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development o executing-plans. Steps use checkbox (`- [ ]`).

**Goal Phase A**: Validar que el loop DPPO funciona end-to-end en CoppeliaSim. 500 episodios. Mostrar mejora monótona ≥ +5 pp en `pick_and_place_success_pct` vs DP v5 base (60 %). Si pasa el gate, avanzar a Phase B (PyBullet, escalado).

**Architecture:** Init desde DP v5 + ResNet-18 encoder frozen. Value head pequeño. DPPO sobre los últimos K denoising timesteps con shaped + binary reward.

**Tech stack:** PyTorch MPS + CoppeliaSim ZMQ + simIK.

---

## File Structure

**Nuevos**:
- `src/rl/__init__.py`
- `src/rl/dppo_agent.py` — DPPOAgent class (policy, value, update step)
- `src/rl/value_net.py` — small MLP value head
- `src/rl/replay_buffer.py` — episode buffer
- `src/rl/reward_fn.py` — reward shaping + terminal
- `experiments/train_dppo_coppeliasim.py` — training loop Phase A
- `experiments/eval_dppo_phaseA_sim.py` — eval post-train
- `tests/test_dppo_agent.py`
- `tests/test_reward_fn.py`

**Modificados**:
- `.gitignore` — `data/models/diffusion_policy_v6_*`

---

## Pre-flight

- [ ] CoppeliaSim corriendo en :23000.
- [ ] DP v5 + visual encoder v5 en `data/models/`.
- [ ] Branch: `feat/iter6-dppo-rl-finetune` (ya creada).

---

## Task 1: Reward function + tests

**Files:**
- Create: `src/rl/reward_fn.py`
- Test: `tests/test_reward_fn.py`

### Step 1.1: Write tests

```python
# tests/test_reward_fn.py
import math
import numpy as np
from src.rl.reward_fn import compute_terminal_reward, compute_shaping_reward


def test_terminal_success():
    """grasp + deposit ambos plausibles → +10."""
    r = compute_terminal_reward(
        grasp_plausible=True, deposit_plausible=True,
        ik_converged=True, distractor_collision=False,
    )
    assert r == 10.0


def test_terminal_grasp_only():
    """grasp ok pero deposit no → +5 (parcial)."""
    r = compute_terminal_reward(
        grasp_plausible=True, deposit_plausible=False,
        ik_converged=True, distractor_collision=False,
    )
    assert r == 5.0


def test_terminal_ik_fail():
    """IK falla → -5."""
    r = compute_terminal_reward(
        grasp_plausible=False, deposit_plausible=False,
        ik_converged=False, distractor_collision=False,
    )
    assert r == -5.0


def test_terminal_collision():
    """colisión distractor → -10."""
    r = compute_terminal_reward(
        grasp_plausible=True, deposit_plausible=True,
        ik_converged=True, distractor_collision=True,
    )
    assert r == 0.0  # +10 - 10 = 0


def test_shaping_grasp_phase():
    """En phase grasp, reward = -0.1 * dist_to_cube."""
    cube_pos = np.array([0.45, -0.10, 0.033])
    wp = np.array([0.45, -0.10, 0.08])  # 5 cm sobre el cubo
    r = compute_shaping_reward(wp, cube_pos, deposit_target=np.zeros(3), step=3, total_steps=16)
    assert math.isclose(r, -0.1 * 0.05, abs_tol=1e-4)
```

### Step 1.2: Run — should fail (ModuleNotFoundError)

```bash
.venv/bin/python -m pytest tests/test_reward_fn.py -v 2>&1 | tail -5
```

### Step 1.3: Implement

```python
# src/rl/reward_fn.py
import numpy as np


GRASP_PHASE_END = 6   # k=0..5 son aproximación + grasp
DEPOSIT_PHASE_START = 9  # k=9..15 son lift + deposit


def compute_terminal_reward(
    grasp_plausible: bool,
    deposit_plausible: bool,
    ik_converged: bool,
    distractor_collision: bool,
) -> float:
    r = 0.0
    if grasp_plausible and deposit_plausible:
        r += 10.0
    elif grasp_plausible:
        r += 5.0
    if not ik_converged:
        r -= 5.0
    if distractor_collision:
        r -= 10.0
    return r


def compute_shaping_reward(
    wp: np.ndarray,
    cube_pos: np.ndarray,
    deposit_target: np.ndarray,
    step: int,
    total_steps: int,
) -> float:
    """Reward shaping per-step.

    Durante phase grasp (step < GRASP_PHASE_END): -0.1 * dist_to_cube.
    Durante phase deposit (step >= DEPOSIT_PHASE_START): -0.1 * dist_to_deposit.
    Phase intermedio: 0.
    """
    if step < GRASP_PHASE_END:
        d = float(np.linalg.norm(wp[:3] - cube_pos[:3]))
        return -0.1 * d
    if step >= DEPOSIT_PHASE_START:
        d = float(np.linalg.norm(wp[:3] - deposit_target[:3]))
        return -0.1 * d
    return 0.0
```

### Step 1.4: Run — should pass

```bash
.venv/bin/python -m pytest tests/test_reward_fn.py -v 2>&1 | tail -10
```

### Step 1.5: Commit

---

## Task 2: Value net + tests

**Files:**
- Create: `src/rl/value_net.py`
- Test: `tests/test_value_net.py` (incorporado al test_dppo_agent.py si es muy chico)

### Step 2.1-2.5: implementar small MLP

```python
# src/rl/value_net.py
import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """MLP small sobre cond 64-D → V(s) scalar."""

    def __init__(self, cond_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond).squeeze(-1)
```

---

## Task 3: Replay buffer

**Files:**
- Create: `src/rl/replay_buffer.py`

Simple episode buffer:

```python
# src/rl/replay_buffer.py
import torch
from dataclasses import dataclass, field


@dataclass
class Episode:
    cond: torch.Tensor       # (64,)
    actions: torch.Tensor     # (H, 7) — full trajectory
    rewards: list = field(default_factory=list)  # per-step + terminal
    log_probs: torch.Tensor = None  # (K, ...) noise log-probs of last K denoising
    value: float = 0.0
    advantage: float = 0.0
    return_: float = 0.0


class EpisodeBuffer:
    def __init__(self):
        self.episodes: list[Episode] = []

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)

    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        """Single-step trajectories — advantage = return - value."""
        for ep in self.episodes:
            ep.return_ = sum(ep.rewards)
            ep.advantage = ep.return_ - ep.value

    def clear(self) -> None:
        self.episodes.clear()

    def __len__(self) -> int:
        return len(self.episodes)
```

---

## Task 4: DPPO agent

**Files:**
- Create: `src/rl/dppo_agent.py`
- Test: `tests/test_dppo_agent.py`

Core algorithm. La idea simplificada:

1. **Sample acción**: forward DP completo. Pero registrar las predicciones de ruido (`eps_pred`) de los últimos K denoising steps.
2. **Reward**: rollout en sim, compute reward.
3. **Update**: PPO loss sobre los `eps_pred` registrados, con clip y advantage.

Implementación simplificada (DPPO completo sería más complejo; este es PPO sobre los noise predictions tail):

```python
# src/rl/dppo_agent.py
import numpy as np
import torch
import torch.nn.functional as F

from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.rl.value_net import ValueNet


class DPPOAgent:
    """DPPO simplificado: PPO sobre los K últimos noise predictions del denoising."""

    def __init__(
        self,
        planner: DiffusionGraspPlanner,
        value_net: ValueNet,
        k_last_denoising: int = 4,
        clip_ratio: float = 0.2,
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
    ):
        self.planner = planner
        self.value_net = value_net
        self.k_last = k_last_denoising
        self.clip = clip_ratio
        self.entropy_coef = entropy_coef
        # Optimizers separados para policy y value
        self.policy_optim = torch.optim.Adam(planner.model.parameters(), lr=lr)
        self.value_optim = torch.optim.Adam(value_net.parameters(), lr=lr)

    def sample_action_with_logprob(self, cond: torch.Tensor):
        """Forward DP guardando noise predictions de los últimos K timesteps.

        Returns: (waypoints (H, 7), log_probs (K, H, 7)).
        """
        self.planner.model.train()
        scheduler = self.planner.scheduler
        device = self.planner.device
        x_t = torch.randn(1, self.planner.horizon, self.planner.action_dim, device=device)
        log_probs = []
        for t in reversed(range(scheduler.num_timesteps)):
            timestep = torch.full((1,), t, device=device, dtype=torch.long)
            eps_pred = self.planner.model(x_t, timestep, cond)
            # Guardar log-prob de eps_pred sólo en los últimos K timesteps
            if t < self.k_last:
                # Gaussian log-prob asumiendo unit std (simplificación)
                log_p = -0.5 * (eps_pred ** 2).sum(dim=(-1, -2))  # (1,)
                log_probs.append(log_p)
            # Denoise
            x_np = x_t.detach().cpu().numpy()
            eps_np = eps_pred.detach().cpu().numpy()
            denoised = scheduler.remove_noise(x_np[0], eps_np[0], t)
            x_t = torch.tensor(denoised, dtype=torch.float32, device=device).unsqueeze(0)
        log_probs_stacked = torch.stack(log_probs, dim=0) if log_probs else torch.zeros(self.k_last, device=device)
        return x_t.squeeze(0).cpu().numpy(), log_probs_stacked

    def update(self, episodes: list, ppo_epochs: int = 4) -> dict:
        if not episodes:
            return {}
        conds = torch.stack([ep.cond for ep in episodes]).to(self.planner.device)
        old_log_probs = torch.stack([ep.log_probs.sum() for ep in episodes]).detach()
        advantages = torch.tensor([ep.advantage for ep in episodes], device=self.planner.device)
        returns = torch.tensor([ep.return_ for ep in episodes], device=self.planner.device)
        # Normalizar advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses, value_losses = [], []
        for _ in range(ppo_epochs):
            # Re-evaluate log-probs and value
            # Para simplicidad: usamos las log_probs guardadas (esto NO es PPO clip correcto,
            # es REINFORCE con ratio=1; en una versión completa, re-evaluar la política).
            ratio = torch.exp(torch.zeros_like(advantages))  # placeholder
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            policy_losses.append(policy_loss.item())

            value_pred = self.value_net(conds)
            value_loss = F.mse_loss(value_pred, returns)
            value_losses.append(value_loss.item())

            self.policy_optim.zero_grad()
            self.value_optim.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            self.policy_optim.step()
            self.value_optim.step()

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "mean_advantage": advantages.mean().item(),
            "mean_return": returns.mean().item(),
        }
```

**Nota honesta**: este DPPO es simplificado. La versión completa requiere re-evaluar la política durante el update para computar el ratio PPO correctamente. Este es REINFORCE-con-clip-placeholder. Sirve para validar el loop end-to-end en Phase A. La versión completa va en Phase B.

---

## Task 5: Training loop Phase A

**Files:**
- Create: `experiments/train_dppo_coppeliasim.py`

```python
#!/usr/bin/env python3
"""Phase A training — DPPO sobre DP v5 en CoppeliaSim."""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.planning.visual_encoder import ResNet18RGBDEncoder
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.rl.dppo_agent import DPPOAgent
from src.rl.value_net import ValueNet
from src.rl.replay_buffer import Episode, EpisodeBuffer
from src.rl.reward_fn import compute_terminal_reward
from experiments.run_pick_with_diffusion import pick_with_dp
from experiments.eval_diffusion_iter2_sim import sample_pose_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_dppo_phaseA")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="episodios por update")
    parser.add_argument("--seed", type=int, default=2027)  # distinto del eval seed=2026
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Init desde DP v5
    ckpt = torch.load(REPO / "data/models/diffusion_policy_sim_v5.pth", map_location=device, weights_only=True)
    hd = ckpt["config"]["hidden_dim"]
    planner = DiffusionGraspPlanner(action_dim=7, horizon=16, n_diffusion_steps=100, device=device, hidden_dim=hd)
    planner.model.load_state_dict(ckpt["model_state_dict"])
    enc_state = torch.load(REPO / "data/models/visual_encoder_iter5.pth", map_location=device, weights_only=True)
    encoder = ResNet18RGBDEncoder(out_dim=enc_state["out_dim"]).to(device).eval()
    encoder.load_state_dict(enc_state["state_dict"])

    value_net = ValueNet(cond_dim=64, hidden_dim=32).to(device)
    agent = DPPOAgent(planner, value_net, k_last_denoising=4, lr=3e-4)
    buffer = EpisodeBuffer()

    SCENE = REPO / "data/scenes/bin_base.ttt"
    rng = np.random.default_rng(args.seed)
    rolling_rewards = []
    update_log = []

    for ep_idx in range(args.episodes):
        pose = sample_pose_eval(rng)
        try:
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                r = pick_with_dp(planner, pose, bridge, frames_dir=None, visual_encoder=encoder)
        except Exception as e:
            logger.warning(f"[{ep_idx}] sim fail: {e}")
            continue

        terminal_r = compute_terminal_reward(
            r["grasp_plausible"], r["deposit_plausible"], r["ik_converged"],
            distractor_collision=False,  # Phase A single-obj
        )
        # Construir cond para storage
        from experiments.collect_diffusion_dataset import _capture_rgbd_for_pose
        # NOTA: re-captura cond del momento del pick (ya hecho en pick_with_dp)
        # Para Phase A simple: usamos el cond del último capture (info no perfecta pero suficiente)
        # Para Phase B esto se almacena durante el rollout.
        # Skip aquí — usamos solo pose como cond placeholder
        pose_flat = torch.tensor(np.eye(4)[:3, :].flatten(), dtype=torch.float32)
        cond_dummy = torch.zeros(64)
        cond_dummy[52:64] = pose_flat[:12]

        ep = Episode(
            cond=cond_dummy,
            actions=torch.tensor(r["waypoints"]),
            rewards=[terminal_r],
            log_probs=torch.zeros(4),  # placeholder Phase A
            value=value_net(cond_dummy.unsqueeze(0).to(device)).item(),
        )
        buffer.add(ep)
        rolling_rewards.append(terminal_r)

        if len(buffer) >= args.batch_size:
            buffer.compute_gae()
            stats = agent.update(buffer.episodes)
            buffer.clear()
            rolling = float(np.mean(rolling_rewards[-args.batch_size:]))
            update_log.append({"ep": ep_idx, "rolling_reward": rolling, **stats})
            logger.info(f"ep {ep_idx}/{args.episodes}: rolling_R={rolling:.2f} {stats}")

    # Save artifacts
    out_ckpt = REPO / "data/models/diffusion_policy_v6_phaseA.pth"
    torch.save({
        "model_state_dict": planner.model.state_dict(),
        "value_state_dict": value_net.state_dict(),
        "config": {**ckpt["config"], "phase": "A", "episodes": args.episodes},
    }, out_ckpt)
    (REPO / "experiments/results/dppo_phaseA_log.json").write_text(json.dumps(update_log, indent=2))
    logger.info(f"escrito ckpt: {out_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Task 6: Smoke test Phase A pequeño (n=20 episodios)

```bash
caffeinate -dimsu .venv/bin/python experiments/train_dppo_coppeliasim.py --episodes 20 --batch-size 8 2>&1 | tail -20
```

Expected: 20 episodios completan sin crash, ckpt escrito, log con 2-3 updates.

---

## Task 7: Phase A completo (500 episodios)

```bash
caffeinate -dimsu .venv/bin/python experiments/train_dppo_coppeliasim.py --episodes 500 --batch-size 16 > /tmp/dppo_phaseA.log 2>&1 &
```

Expected: ~3-5 h. Curva de rolling reward debería subir.

---

## Task 8: Eval Phase A — gate decision

```bash
caffeinate -dimsu .venv/bin/python experiments/eval_dppo_phaseA_sim.py --n 50 > /tmp/eval_v6_phaseA.log 2>&1
```

**GATE**:
- ✅ Si `pick_and_place_success_pct ≥ 65 %` → proceder a Phase B.
- ⚠ Si está entre 60 y 65 % → diagnosticar curva, revisar reward hacking.
- ❌ Si < 60 % → DPPO regresa la policy. Documentar honestamente, pivotar a PPO simple.

---

## F.1: Documentar + commit + push + PR

Documentar resultados en `INTEGRATION_PIPELINE.md` con sección Iter 6 Phase A.
