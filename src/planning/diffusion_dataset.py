"""Dataset PyTorch para fine-tune de la Diffusion Policy sobre datos del sim."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset


class SimPickDataset(Dataset):
    """Dataset de pares (cond, trajectory) para training de DiffusionPolicy.

    Soporta dos formatos:
    - v1/v2: dict {conds: (N, 64), trajs: (N, 16, 7)}.
    - v3:    dict {visual_emb: (N, 52), poses: (N, 16), trajs: (N, 16, 7)}.
             cond = concat[visual_emb (52), poses[:, :12]] -> (N, 64).
    """

    def __init__(self, pt_path):
        pt_path = Path(pt_path)
        data = torch.load(pt_path, weights_only=True)
        if "visual_emb" in data:
            visual_emb = data["visual_emb"].to(torch.float32)
            poses_flat = data["poses"].to(torch.float32)
            pose12 = poses_flat[:, :12]
            self.conds = torch.cat([visual_emb, pose12], dim=1)
        else:
            self.conds = data["conds"].to(torch.float32)
        self.trajs = data["trajs"].to(torch.float32)
        assert self.conds.shape[0] == self.trajs.shape[0], (
            f"mismatched len: conds={self.conds.shape}, trajs={self.trajs.shape}"
        )

    def __len__(self) -> int:
        return self.conds.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conds[i], self.trajs[i]
