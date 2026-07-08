"""Tests para src/perception/checkpoint_adapter.py y pose_estimator.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def fp_checkpoint():
    p = REPO / "experiments/checkpoints/fp_ycbv_checkpoint.json"
    if not p.exists():
        pytest.skip("FP YCB-V checkpoint no disponible")
    return p


class TestPoseEstimatorProtocol:
    def test_pose_estimate_dataclass(self):
        from src.perception.pose_estimator import PoseEstimate
        R = np.eye(3)
        t = np.array([100, 200, 800])
        est = PoseEstimate(R=R, t=t, confidence=0.95, inference_time_s=0.05, method="fp")
        assert est.method == "fp"
        assert est.confidence == 0.95

    def test_protocol_compatibility(self):
        from src.perception.checkpoint_adapter import CheckpointPoseEstimator
        from src.perception.pose_estimator import PoseEstimator
        # Verifica que CheckpointPoseEstimator cumple el protocolo (runtime_checkable)
        # Nota: necesitamos una instancia para verificar
        ckpt_path = REPO / "experiments/checkpoints/fp_ycbv_checkpoint.json"
        if not ckpt_path.exists():
            pytest.skip("Checkpoint YCB-V no disponible")
        est = CheckpointPoseEstimator(ckpt_path, method="foundationpose")
        # Runtime protocol check
        assert isinstance(est, PoseEstimator)


class TestCheckpointAdapter:
    def test_loads_fp_checkpoint(self, fp_checkpoint):
        from src.perception.checkpoint_adapter import CheckpointPoseEstimator
        est = CheckpointPoseEstimator(fp_checkpoint, method="foundationpose")
        assert est.name == "foundationpose"
        assert "NC" in est.license
        assert est.is_commercializable() is False

    def test_unknown_method_raises(self, fp_checkpoint):
        from src.perception.checkpoint_adapter import CheckpointPoseEstimator
        with pytest.raises(ValueError, match="desconocido"):
            CheckpointPoseEstimator(fp_checkpoint, method="not_a_method")

    def test_fp_returns_same_pose_no_noise(self, fp_checkpoint):
        """Sin ruido (foundationpose), el adapter devuelve la pose del checkpoint exacta."""
        from src.perception.checkpoint_adapter import CheckpointPoseEstimator
        est1 = CheckpointPoseEstimator(fp_checkpoint, method="foundationpose", seed=42)
        est2 = CheckpointPoseEstimator(fp_checkpoint, method="foundationpose", seed=43)
        # Tomar la primera escena/img/obj del checkpoint
        first_key = next(iter(est1._index.keys()))
        scene, img, obj, gt = first_key
        pose1 = est1.predict_pose(scene_id=scene, img_id=img, obj_id=obj, gt_idx=gt)
        pose2 = est2.predict_pose(scene_id=scene, img_id=img, obj_id=obj, gt_idx=gt)
        # Sin ruido, las dos llamadas con seeds distintas devuelven lo mismo
        np.testing.assert_allclose(pose1.R, pose2.R)
        np.testing.assert_allclose(pose1.t, pose2.t)

    def test_noise_changes_pose(self, fp_checkpoint):
        """Con ruido (freezev2), el adapter devuelve poses cercanas pero distintas."""
        from src.perception.checkpoint_adapter import CheckpointPoseEstimator
        est_clean = CheckpointPoseEstimator(fp_checkpoint, method="foundationpose")
        est_noisy = CheckpointPoseEstimator(fp_checkpoint, method="freezev2", seed=42)
        first_key = next(iter(est_clean._index.keys()))
        scene, img, obj, gt = first_key
        clean = est_clean.predict_pose(scene_id=scene, img_id=img, obj_id=obj, gt_idx=gt)
        noisy = est_noisy.predict_pose(scene_id=scene, img_id=img, obj_id=obj, gt_idx=gt)
        # La pose noisy debe estar cerca pero no identica
        delta_t = np.linalg.norm(clean.t - noisy.t)
        assert delta_t > 0  # hay ruido
        assert delta_t < 20  # ruido razonable (FreeZeV2 ~3mm std, max ~10mm)

    def test_commercializable_flags(self, fp_checkpoint):
        from src.perception.checkpoint_adapter import CheckpointPoseEstimator
        # foundationpose: NC, no commercializable
        assert CheckpointPoseEstimator(fp_checkpoint, method="foundationpose").is_commercializable() is False
        # freezev2: Apache-2.0, commercializable
        assert CheckpointPoseEstimator(fp_checkpoint, method="freezev2").is_commercializable() is True
        # any6d: MIT, commercializable
        assert CheckpointPoseEstimator(fp_checkpoint, method="any6d").is_commercializable() is True
        # megapose: AGPL-3.0, no commercializable (copyleft viral)
        assert CheckpointPoseEstimator(fp_checkpoint, method="megapose").is_commercializable() is False

    def test_units_normalized_to_mm(self, fp_checkpoint):
        """Las poses FP suelen venir en metros (<5); el adapter las pasa a mm."""
        from src.perception.checkpoint_adapter import CheckpointPoseEstimator
        est = CheckpointPoseEstimator(fp_checkpoint, method="foundationpose")
        first_key = next(iter(est._index.keys()))
        scene, img, obj, gt = first_key
        pose = est.predict_pose(scene_id=scene, img_id=img, obj_id=obj, gt_idx=gt)
        # Translation deberia estar en mm (norm > 100 tipicamente para escenas BOP)
        assert np.linalg.norm(pose.t) > 100, f"|t|={np.linalg.norm(pose.t)} parece estar en metros, no mm"


class TestListAvailableMethods:
    def test_returns_all_methods(self):
        from src.perception.checkpoint_adapter import list_available_methods
        methods = list_available_methods()
        expected = {"foundationpose", "freezev2", "megapose", "any6d", "sampose"}
        assert expected.issubset(set(methods.keys()))

    def test_each_method_has_required_fields(self):
        from src.perception.checkpoint_adapter import list_available_methods
        methods = list_available_methods()
        required = {"license", "commercial", "noise_t_mm_std", "noise_R_rad_std", "reference"}
        for name, profile in methods.items():
            assert required.issubset(profile.keys()), f"{name} falta campos"


class TestExp15ResultsArePlausible:
    """Sanity check sobre el JSON de exp15 si esta commiteado."""

    def test_exp15_json_present_and_valid(self):
        import json
        p = REPO / "experiments/results/exp15_open_license/exp15_results.json"
        if not p.exists():
            pytest.skip("exp15 results no commiteado todavia")
        d = json.loads(p.read_text())
        assert "datasets" in d
        # Foundationpose siempre debe tener AUC > 0.85 sobre YCB-V
        if "ycbv" in d["datasets"] and "foundationpose" in d["datasets"]["ycbv"]:
            fp = d["datasets"]["ycbv"]["foundationpose"]
            assert fp["auc_adds_50mm"]["point"] > 0.85
