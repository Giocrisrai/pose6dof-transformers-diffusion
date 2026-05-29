from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNet18RGBDEncoder(nn.Module):
    """ResNet-18 pretrained on ImageNet, patched to accept 4-channel RGB-D.

    Conv1 RGB weights are copied from the ImageNet checkpoint and the depth
    channel is initialized to zero. All backbone params are frozen; only the
    final Linear head is trainable.
    """

    def __init__(self, out_dim: int = 52):
        super().__init__()
        weights = tvm.ResNet18_Weights.DEFAULT
        backbone = tvm.resnet18(weights=weights)

        original_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = original_conv1.weight
            new_conv1.weight[:, 3, :, :] = 0.0
        backbone.conv1 = new_conv1

        backbone.fc = nn.Identity()

        for p in backbone.parameters():
            p.requires_grad = False

        self.backbone = backbone
        self.head = nn.Linear(512, out_dim)

    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(rgbd)
        return self.head(feats)
