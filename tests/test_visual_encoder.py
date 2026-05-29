import torch

from src.planning.visual_encoder import ResNet18RGBDEncoder


def test_output_shape():
    enc = ResNet18RGBDEncoder(out_dim=52)
    rgbd = torch.zeros(2, 4, 224, 224)
    emb = enc(rgbd)
    assert emb.shape == (2, 52)


def test_resnet_frozen():
    enc = ResNet18RGBDEncoder(out_dim=52)
    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    total = sum(p.numel() for p in enc.parameters())
    assert trainable < total * 0.05, f"too many trainable: {trainable}/{total}"
    assert enc.head.weight.requires_grad


def test_conv1_4channels_init():
    enc = ResNet18RGBDEncoder(out_dim=52)
    assert enc.backbone.conv1.in_channels == 4
    w = enc.backbone.conv1.weight.data
    assert torch.allclose(w[:, 3, :, :], torch.zeros_like(w[:, 3, :, :]))
    assert not torch.allclose(w[:, :3, :, :], torch.zeros_like(w[:, :3, :, :]))
