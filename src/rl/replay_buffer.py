"""Simple episode buffer for DPPO (Phase A)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class Episode:
    cond: torch.Tensor
    actions: torch.Tensor
    rewards: List[float] = field(default_factory=list)
    log_probs: torch.Tensor = None  # set during sample
    value: float = 0.0
    advantage: float = 0.0
    return_: float = 0.0


class EpisodeBuffer:
    def __init__(self) -> None:
        self.episodes: List[Episode] = []

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)

    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        """Single-step trajectories (1 reward terminal per episode).

        Advantage = return - value (TD(0) style at end of trajectory).
        """
        for ep in self.episodes:
            ep.return_ = float(sum(ep.rewards))
            ep.advantage = ep.return_ - ep.value

    def clear(self) -> None:
        self.episodes.clear()

    def __len__(self) -> int:
        return len(self.episodes)
