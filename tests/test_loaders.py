"""Tests para src/utils/loaders.py."""
from pathlib import Path

import numpy as np
import pytest

from src.utils.loaders import (
    DATASET_INFO,
    PoseSample,
    _normalize_translation,
    load_predictions_with_gt,
)


class TestPoseSample:
    def test_creation(self):
        s = PoseSample(
            scene_id="000001", img_id=1, obj_id=5,
            R_pred=np.eye(3), t_pred=np.array([1.0, 2.0, 3.0]),
            R_gt=np.eye(3), t_gt=np.array([1.1, 2.0, 3.0]),
            points=np.zeros((100, 3)),
        )
        assert s.scene_id == "000001"
        assert s.obj_id == 5
        assert s.points.shape == (100, 3)
        assert s.extra == {}

    def test_with_extra(self):
        s = PoseSample(
            scene_id="000001", img_id=1, obj_id=5,
            R_pred=np.eye(3), t_pred=np.zeros(3),
            R_gt=np.eye(3), t_gt=np.zeros(3),
            points=np.zeros((10, 3)),
            extra={"gt_idx": 2},
        )
        assert s.extra["gt_idx"] == 2


class TestDatasetInfo:
    def test_known_datasets(self):
        assert "ycbv" in DATASET_INFO
        assert "tless" in DATASET_INFO

    def test_keys_present(self):
        for ds, info in DATASET_INFO.items():
            assert "split" in info
            assert "checkpoint" in info
            assert "data_path" in info


class TestNormalizeTranslation:
    def test_meters_to_mm(self):
        # Vector con norma < 5 (metros)
        t = np.array([0.1, 0.2, 0.3])
        out = _normalize_translation(t)
        assert np.allclose(out, t * 1000.0)

    def test_already_mm(self):
        # Vector con norma > 5 (mm)
        t = np.array([100.0, 200.0, 300.0])
        out = _normalize_translation(t)
        assert np.allclose(out, t)

    def test_boundary(self):
        # En el limite (norma = 5)
        t = np.array([3.0, 4.0, 0.0])  # norma = 5.0
        out = _normalize_translation(t)
        # >=5 no convierte
        assert np.allclose(out, t)


class TestLoadPredictionsWithGT:
    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Dataset desconocido"):
            load_predictions_with_gt("invalid_dataset")

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[1] / "experiments/checkpoints/fp_ycbv_checkpoint.json").exists(),
        reason="Checkpoint no descargado"
    )
    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[1] / "data/datasets/ycbv").exists(),
        reason="Dataset YCB-V no descargado"
    )
    def test_load_ycbv_small_sample(self):
        samples = load_predictions_with_gt("ycbv", n_max=5)
        assert len(samples) > 0
        # Verificar cada sample
        for s in samples:
            assert isinstance(s, PoseSample)
            assert s.R_pred.shape == (3, 3)
            assert s.t_pred.shape == (3,)
            assert s.R_gt.shape == (3, 3)
            assert s.t_gt.shape == (3,)
            assert s.points.ndim == 2
            assert s.points.shape[1] == 3
            assert s.fp_time_ms > 0
            # Traslacion en mm (norma > 100 para escenas YCB-V)
            assert np.linalg.norm(s.t_pred) > 100
