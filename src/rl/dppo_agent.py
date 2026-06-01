"""DPPO agent — simplified PPO sobre la cadena de denoising (Phase A PoC).

NOTA HONESTA: esta es una version SIMPLIFICADA para validar el loop end-to-end.
- PPO ratio se computa con log-probs almacenados al sample (no se re-evalua durante update),
  asi que en cada update inicial el ratio es ~1.0. Sirve como REINFORCE-con-clip.
- En Phase B se implementa DPPO completo: re-evaluacion + ratio real + clip efectivo.

Referencia full DPPO: Ren et al. NeurIPS 2024, https://github.com/irom-lab/dppo
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.rl.replay_buffer import Episode
from src.rl.value_net import ValueNet


class DPPOAgent:
    def __init__(
        self,
        planner: DiffusionGraspPlanner,
        value_net: ValueNet,
        k_last_denoising: int = 4,
        clip_ratio: float = 0.2,
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        kl_coef: float = 1.0,
        ref_model=None,
    ):
        self.planner = planner
        self.value_net = value_net
        self.k_last = k_last_denoising
        self.clip = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.kl_coef = kl_coef
        self.ref_model = ref_model  # frozen reference policy (v5) for KD anchor
        if self.ref_model is not None:
            for p in self.ref_model.parameters():
                p.requires_grad = False
            self.ref_model.eval()
        self.policy_optim = torch.optim.Adam(planner.model.parameters(), lr=lr)
        self.value_optim = torch.optim.Adam(value_net.parameters(), lr=lr)

    @torch.no_grad()
    def sample_action_with_logprob(self, cond: torch.Tensor):
        """Forward diffusion guardando log-prob (Gaussian unit-std approx) de los ultimos K timesteps."""
        self.planner.model.eval()
        scheduler = self.planner.scheduler
        device = self.planner.device
        x_t = torch.randn(1, self.planner.horizon, self.planner.action_dim, device=device)
        log_probs = []
        for t in reversed(range(scheduler.num_timesteps)):
            timestep = torch.full((1,), t, device=device, dtype=torch.long)
            eps_pred = self.planner.model(x_t, timestep, cond)
            if t < self.k_last:
                log_p = -0.5 * (eps_pred ** 2).sum()
                log_probs.append(log_p)
            x_np = x_t.cpu().numpy()
            eps_np = eps_pred.cpu().numpy()
            denoised = scheduler.remove_noise(x_np[0], eps_np[0], t)
            x_t = torch.tensor(denoised, dtype=torch.float32, device=device).unsqueeze(0)
        log_probs_stacked = (
            torch.stack(log_probs, dim=0) if log_probs else torch.zeros(self.k_last, device=device)
        )
        return x_t.squeeze(0).cpu().numpy(), log_probs_stacked.detach()

    def update(self, episodes: List[Episode], ppo_epochs: int = 4) -> dict:
        """Self-imitation learning weighted by advantage (Phase A PoC).

        Para cada episodio con advantage > 0, hacer BC weighted: re-entrenar
        la noise net en una iter de DDPM forward con la trayectoria del
        episodio como target. Episodios con advantage <= 0 se ignoran
        (no aportan signal positiva).

        Para PPO completo necesitamos re-evaluar log_prob bajo politica
        actual; eso va en Phase B.
        """
        if not episodes:
            return {}
        device = self.planner.device
        scheduler = self.planner.scheduler
        n_timesteps = scheduler.num_timesteps

        advantages_raw = torch.tensor([ep.advantage for ep in episodes], device=device)
        returns = torch.tensor([ep.return_ for ep in episodes], device=device)
        # Normalize
        adv_norm = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)

        # Solo entrenar con advantage positiva (self-imitation)
        positive_mask = adv_norm > 0
        positive_eps = [ep for ep, m in zip(episodes, positive_mask) if m]
        positive_weights = adv_norm[positive_mask]

        policy_losses, value_losses = [], []
        for _ in range(ppo_epochs):
            # Value update sobre TODO el batch
            conds_all = torch.stack([ep.cond for ep in episodes]).to(device)
            value_pred = self.value_net(conds_all)
            value_loss = F.mse_loss(value_pred, returns)
            value_losses.append(value_loss.item())
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()

            # Policy update: BC sobre positivos + KL/KD a referencia v5 sobre TODO
            self.planner.model.train()
            # BC weighted (signal de RL) sobre positivos
            bc_loss = torch.tensor(0.0, device=device)
            if positive_eps:
                conds_pos = torch.stack([ep.cond for ep in positive_eps]).to(device)
                actions_pos = torch.stack([ep.actions for ep in positive_eps]).to(device).float()
                B = actions_pos.shape[0]
                t = torch.randint(0, n_timesteps, (B,), device=device, dtype=torch.long)
                noisy, true_noise = scheduler.add_noise_batch(actions_pos, t)
                pred_noise = self.planner.model(noisy, t, conds_pos)
                per_sample = ((pred_noise - true_noise) ** 2).mean(dim=(-1, -2))
                bc_loss = (per_sample * positive_weights).mean()
            # KD term: anclar a la referencia v5 con noisy actions del batch ENTERO
            kd_loss = torch.tensor(0.0, device=device)
            if self.ref_model is not None:
                conds_all_pol = torch.stack([ep.cond for ep in episodes]).to(device)
                actions_all = torch.stack([ep.actions for ep in episodes]).to(device).float()
                Ba = actions_all.shape[0]
                ta = torch.randint(0, n_timesteps, (Ba,), device=device, dtype=torch.long)
                noisy_a, _ = scheduler.add_noise_batch(actions_all, ta)
                pred_new = self.planner.model(noisy_a, ta, conds_all_pol)
                with torch.no_grad():
                    pred_ref = self.ref_model(noisy_a, ta, conds_all_pol)
                kd_loss = F.mse_loss(pred_new, pred_ref)
            total_policy_loss = bc_loss + self.kl_coef * kd_loss
            policy_losses.append(float(total_policy_loss.item()))
            self.policy_optim.zero_grad()
            total_policy_loss.backward()
            self.policy_optim.step()
            self.planner.model.eval()

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "n_positive_episodes": int(positive_mask.sum().item()),
            "mean_advantage_raw": float(advantages_raw.mean().item()),
            "mean_return": float(returns.mean().item()),
        }
