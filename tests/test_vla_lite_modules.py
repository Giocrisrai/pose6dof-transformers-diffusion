"""Tests para los modulos VLA-lite del exp16:
- CLIPProjector (512 -> 32 D)
- TextGroundedGate (selecciona objeto A vs B a partir de CLIP + RGB)
- assemble_cond / assemble_cond_with_selected_pos

No requieren CLIP weights ni training, solo verifican la mecanica.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


@pytest.fixture(scope="module")
def exp16_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp16", REPO / "experiments/exp16_vla_lite_clip.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestCLIPProjector:
    def test_shape(self, exp16_module):
        proj = exp16_module.CLIPProjector(in_dim=512, out_dim=32)
        x = torch.randn(7, 512)
        y = proj(x)
        assert y.shape == (7, 32)

    def test_differentiable(self, exp16_module):
        proj = exp16_module.CLIPProjector()
        x = torch.randn(4, 512, requires_grad=False)
        y = proj(x)
        y.sum().backward()
        # Algun parametro debe tener gradiente
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in proj.parameters())


class TestTextGroundedGate:
    def test_shape_and_softmax(self, exp16_module):
        gate = exp16_module.TextGroundedGate(clip_dim=512)
        clip = torch.randn(5, 512)
        rgb_a = torch.rand(5, 3)
        rgb_b = torch.rand(5, 3)
        g_a, g_b = gate(clip, rgb_a, rgb_b)
        # Shapes correctos
        assert g_a.shape == (5,) and g_b.shape == (5,)
        # Suma = 1 (softmax)
        torch.testing.assert_close(g_a + g_b, torch.ones(5), atol=1e-5, rtol=1e-5)

    def test_in_range(self, exp16_module):
        gate = exp16_module.TextGroundedGate(clip_dim=512)
        clip = torch.randn(10, 512)
        rgb_a = torch.rand(10, 3)
        rgb_b = torch.rand(10, 3)
        g_a, g_b = gate(clip, rgb_a, rgb_b)
        assert (g_a >= 0).all() and (g_a <= 1).all()
        assert (g_b >= 0).all() and (g_b <= 1).all()

    def test_differentiable(self, exp16_module):
        gate = exp16_module.TextGroundedGate(clip_dim=512)
        clip = torch.randn(3, 512)
        rgb_a = torch.rand(3, 3)
        rgb_b = torch.rand(3, 3)
        g_a, g_b = gate(clip, rgb_a, rgb_b)
        loss = (g_a - 1).pow(2).mean() + g_b.pow(2).mean()  # forzar elegir A
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters())


class TestAssembleCond:
    def test_assemble_cond_basic(self, exp16_module):
        sc = torch.randn(4, 64)
        clip_proj = torch.randn(4, 32)
        cb = exp16_module.assemble_cond(sc, clip_proj)
        assert cb.shape == (4, 64)
        # Las dims [19..51] deben coincidir con clip_proj
        torch.testing.assert_close(cb[:, 19:51], clip_proj)
        # El resto debe seguir igual que sc
        torch.testing.assert_close(cb[:, :19], sc[:, :19])
        torch.testing.assert_close(cb[:, 51:], sc[:, 51:])

    def test_assemble_cond_with_selected_pos(self, exp16_module):
        sc = torch.randn(2, 64)
        selected_pos = torch.tensor([[0.1, 0.2, 0.8], [-0.3, 0.0, 0.9]])
        cb = exp16_module.assemble_cond_with_selected_pos(sc, selected_pos)
        assert cb.shape == (2, 64)
        # [0..3] reemplazado por selected_pos
        torch.testing.assert_close(cb[:, :3], selected_pos)
        # Resto inalterado
        torch.testing.assert_close(cb[:, 3:], sc[:, 3:])


class TestGenerateScene:
    def test_yields_valid_scene(self, exp16_module):
        rng = np.random.default_rng(0)
        scene = exp16_module.generate_scene(rng)
        # Campos obligatorios
        for key in ["p_a", "p_b", "c_a", "c_b", "target_idx", "target_pos",
                    "distractor_pos", "text", "traj_gt"]:
            assert key in scene, f"falta {key}"
        # Colores distintos
        assert scene["c_a"] != scene["c_b"]
        # Target coherente con target_idx
        if scene["target_idx"] == 0:
            np.testing.assert_allclose(scene["target_pos"], scene["p_a"])
        else:
            np.testing.assert_allclose(scene["target_pos"], scene["p_b"])
        # Trayectoria forma correcta
        assert scene["traj_gt"].shape == (16, 7)
        # Texto menciona el color target
        target_color = scene["c_a"] if scene["target_idx"] == 0 else scene["c_b"]
        assert target_color in scene["text"]


class TestExp16ResultsAreValid:
    def test_results_pass_criteria(self):
        import json
        p = REPO / "experiments/results/exp16_vla_lite/exp16_results.json"
        if not p.exists():
            pytest.skip("exp16 no ejecutado todavia")
        d = json.loads(p.read_text())
        # Selection accuracy debe superar el criterio
        assert d["evaluation"]["selection_accuracy"] >= 0.75
        assert d["all_criteria_pass"] is True
