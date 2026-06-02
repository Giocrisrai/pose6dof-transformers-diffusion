"""DPPO agent — proper PPO with re-evaluated log-probs over diffusion chain.

Phase B implementation: trata los ultimos K denoising steps como MDP donde
cada accion es el noise prediction. Sample con exploration noise, store
(x_t, t, cond, eps_sampled, log_prob_old). Durante update: re-run model y
computa ratio PPO clip.

Reference: Ren et al., NeurIPS 2024, https://github.com/irom-lab/dppo
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.rl.replay_buffer import Episode
from src.rl.value_net import ValueNet


# Constant exploration variance. sigma=0.5 mantiene ratio en rango razonable
# frente a steps de gradiente: si shift es 0.5 (=sigma), ratio ~ exp(-0.5) = 0.6.
# Con sigma=0.1, shift de 0.5 daba ratio ~ 4e-6 (todo clipped).
SAMPLING_SIGMA = 0.5


def _gaussian_log_prob(x: torch.Tensor, mean: torch.Tensor, sigma: float) -> torch.Tensor:
    """Pseudo log-prob N(x | mean, sigma^2 I) PROMEDIADO sobre event dims.

    NOTA: usar mean en vez de sum es una practica estandar en RL continuo
    para action spaces de alta dimension (Schulman et al. 2017). Sin esto,
    para spaces de 112 dims (16x7), una diferencia de O(sigma) por dim
    produce ratio ~ exp(-56) = 0 → PPO inestable. Con mean, escala bien.
    """
    var = sigma ** 2
    log_norm = -0.5 * math.log(2 * math.pi * var)
    return (-0.5 * (x - mean) ** 2 / var + log_norm).mean(dim=(-1, -2))


@dataclass
class DenoisingStep:
    """Almacena un step del MDP de denoising para re-evaluacion durante update."""
    x_t: torch.Tensor       # (H, action_dim) noisy action at this step
    t: int                  # diffusion timestep
    eps_sampled: torch.Tensor  # (H, action_dim) sampled noise
    log_prob_old: float     # log prob bajo politica al sample time


@dataclass
class DPPOEpisode(Episode):
    """Episode extendido con buffer de denoising steps para PPO update."""
    denoising_steps: List[DenoisingStep] = field(default_factory=list)


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
        kl_coef: float = 0.1,
        ref_model=None,
        sampling_sigma: float = SAMPLING_SIGMA,
    ):
        self.planner = planner
        self.value_net = value_net
        self.k_last = k_last_denoising
        self.clip = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.kl_coef = kl_coef
        self.sigma = sampling_sigma
        self.ref_model = ref_model
        if self.ref_model is not None:
            for p in self.ref_model.parameters():
                p.requires_grad = False
            self.ref_model.eval()
        self.policy_optim = torch.optim.Adam(planner.model.parameters(), lr=lr)
        self.value_optim = torch.optim.Adam(value_net.parameters(), lr=lr)

    @torch.no_grad()
    def sample_action_with_steps(self, cond: torch.Tensor):
        """Forward diffusion con exploration noise en los ultimos K timesteps.

        Returns: (waypoints (H, action_dim), denoising_steps list).
        """
        self.planner.model.eval()
        scheduler = self.planner.scheduler
        device = self.planner.device
        x_t = torch.randn(1, self.planner.horizon, self.planner.action_dim, device=device)
        steps: List[DenoisingStep] = []
        for t in reversed(range(scheduler.num_timesteps)):
            timestep = torch.full((1,), t, device=device, dtype=torch.long)
            eps_pred = self.planner.model(x_t, timestep, cond)
            if t < self.k_last:
                # Sample con exploration: eps = eps_pred + sigma * N(0, I)
                exploration = torch.randn_like(eps_pred) * self.sigma
                eps_sampled = eps_pred + exploration
                # log prob bajo politica
                lp = float(_gaussian_log_prob(eps_sampled, eps_pred, self.sigma).item())
                steps.append(DenoisingStep(
                    x_t=x_t.squeeze(0).cpu().clone(),
                    t=int(t),
                    eps_sampled=eps_sampled.squeeze(0).cpu().clone(),
                    log_prob_old=lp,
                ))
            else:
                eps_sampled = eps_pred
            # Denoise
            x_np = x_t.cpu().numpy()
            eps_np = eps_sampled.cpu().numpy()
            denoised = scheduler.remove_noise(x_np[0], eps_np[0], t)
            x_t = torch.tensor(denoised, dtype=torch.float32, device=device).unsqueeze(0)
        return x_t.squeeze(0).cpu().numpy(), steps

    def update(self, episodes: List[DPPOEpisode], ppo_epochs: int = 4) -> dict:
        """PPO update sobre los stored denoising steps + KL anchor opcional."""
        if not episodes:
            return {}
        device = self.planner.device
        # Aplanar todos los denoising steps de todos los episodios
        all_xt = []
        all_t = []
        all_cond = []
        all_eps_sampled = []
        all_log_prob_old = []
        all_advantage = []
        for ep in episodes:
            for ds in ep.denoising_steps:
                all_xt.append(ds.x_t)
                all_t.append(ds.t)
                all_eps_sampled.append(ds.eps_sampled)
                all_log_prob_old.append(ds.log_prob_old)
                all_cond.append(ep.cond)
                all_advantage.append(ep.advantage)
        if not all_xt:
            return {"policy_loss": 0.0, "value_loss": 0.0, "n_steps": 0}

        x_t_batch = torch.stack(all_xt).to(device)
        t_batch = torch.tensor(all_t, dtype=torch.long, device=device)
        cond_batch = torch.stack(all_cond).to(device)
        eps_sampled_batch = torch.stack(all_eps_sampled).to(device)
        log_prob_old = torch.tensor(all_log_prob_old, device=device)
        advantages = torch.tensor(all_advantage, device=device)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Value targets (returns)
        returns = torch.tensor([ep.return_ for ep in episodes], device=device)
        conds_per_episode = torch.stack([ep.cond for ep in episodes]).to(device)

        policy_losses, value_losses, kl_terms, clip_fractions = [], [], [], []
        for _ in range(ppo_epochs):
            # POLICY: re-evaluate noise predictions con el modelo actual
            self.planner.model.train()
            eps_pred_new = self.planner.model(x_t_batch, t_batch, cond_batch)
            log_prob_new = _gaussian_log_prob(eps_sampled_batch, eps_pred_new, self.sigma)
            ratio = torch.exp(log_prob_new - log_prob_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            clip_frac = ((ratio < 1 - self.clip) | (ratio > 1 + self.clip)).float().mean()

            # KL anchor opcional a referencia
            kl_term = torch.tensor(0.0, device=device)
            if self.ref_model is not None:
                with torch.no_grad():
                    eps_pred_ref = self.ref_model(x_t_batch, t_batch, cond_batch)
                kl_term = F.mse_loss(eps_pred_new, eps_pred_ref)

            # VALUE update
            value_pred = self.value_net(conds_per_episode)
            value_loss = F.mse_loss(value_pred, returns)

            total = policy_loss + self.value_loss_coef * value_loss + self.kl_coef * kl_term
            self.policy_optim.zero_grad()
            self.value_optim.zero_grad()
            total.backward()
            self.policy_optim.step()
            self.value_optim.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            kl_terms.append(float(kl_term.item()))
            clip_fractions.append(float(clip_frac.item()))

            self.planner.model.eval()

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "kl_term": float(np.mean(kl_terms)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "mean_advantage": float(advantages.mean().item()),
            "mean_return": float(returns.mean().item()),
            "n_steps": len(all_xt),
            "mean_ratio": float(torch.exp(log_prob_new - log_prob_old).mean().item()),
        }
