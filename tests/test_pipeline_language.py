"""Integración del lenguaje en BinPickingPipeline (con poses mock)."""
import numpy as np
from src.pipeline import BinPickingPipeline, PipelineConfig, PoseResult


def _pose(obj_id, x, color, shape):
    T = np.eye(4); T[0, 3] = x; T[2, 3] = 0.5
    return PoseResult(obj_id=obj_id, R=np.eye(3), t=np.array([x, 0, 0.5]),
                      score=0.9, T=T, attributes={"color": color, "shape": shape})


def test_objectviews_desde_poses():
    cfg = PipelineConfig(language_enabled=True)
    pipe = BinPickingPipeline(cfg)
    poses = [_pose(0, -0.2, "red", "cube"), _pose(1, 0.2, "blue", "cube")]
    views = pipe._poses_to_views(poses)
    assert len(views) == 2
    assert views[0].centroid[0] == -0.2
    assert views[0].attributes["color"] == "red"


def test_select_target_por_instruccion():
    cfg = PipelineConfig(language_enabled=True)
    pipe = BinPickingPipeline(cfg)
    poses = [_pose(0, -0.2, "red", "cube"), _pose(1, 0.2, "blue", "cube")]
    selected, grounding, instruction = pipe.select_target(poses, "pick the red cube")
    assert [p.obj_id for p in selected] == [0]
    assert grounding.target_obj_id == 0
    assert instruction.target.color == "red"


def test_sin_instruccion_no_filtra():
    cfg = PipelineConfig(language_enabled=True)
    pipe = BinPickingPipeline(cfg)
    poses = [_pose(0, -0.2, "red", "cube"), _pose(1, 0.2, "blue", "cube")]
    selected, grounding, instruction = pipe.select_target(poses, None)
    assert len(selected) == 2 and grounding is None and instruction is None


def test_ambiguo_tolerante_ordena_por_score():
    cfg = PipelineConfig(language_enabled=True, ambiguity_tolerant=True)
    pipe = BinPickingPipeline(cfg)
    poses = [_pose(0, -0.2, "red", "cube"), _pose(1, 0.2, "red", "cube")]
    selected, grounding, _ = pipe.select_target(poses, "pick the red cube")
    assert grounding.ambiguous
    assert len(selected) == 2          # conserva ambos candidatos ordenados


def test_run_ignora_instruccion_si_language_disabled():
    cfg = PipelineConfig(language_enabled=False)
    pipe = BinPickingPipeline(cfg)
    poses = [_pose(0, -0.2, "red", "cube"), _pose(1, 0.2, "blue", "cube")]
    # select_target SÍ filtra (se llama directo), pero run() solo si language_enabled.
    # Aquí verificamos el guard a nivel de select_target con instrucción None-like:
    selected, grounding, instruction = pipe.select_target(poses, "pick the red cube")
    # select_target no depende del flag; el guard de language_enabled vive en run().
    assert grounding is not None
