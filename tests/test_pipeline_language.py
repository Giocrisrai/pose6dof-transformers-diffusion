"""Integración del lenguaje en BinPickingPipeline (con poses mock)."""
import importlib
import sys
from pathlib import Path

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


def test_select_target_sin_match_devuelve_vacio():
    cfg = PipelineConfig(language_enabled=True)
    pipe = BinPickingPipeline(cfg)
    poses = [_pose(0, -0.2, "red", "cube"), _pose(1, 0.2, "blue", "cube")]
    selected, grounding, instr = pipe.select_target(poses, "pick the green cylinder")
    assert selected == []
    assert grounding.target_obj_id is None


def test_run_con_lenguaje_filtra_target(monkeypatch):
    cfg = PipelineConfig(language_enabled=True)
    pipe = BinPickingPipeline(cfg)
    pipe._initialized = True
    poses = [_pose(0, -0.2, "red", "cube"), _pose(1, 0.2, "blue", "cube")]
    monkeypatch.setattr(pipe, "estimate_poses", lambda *a, **k: list(poses))
    captured = {}
    def fake_plan(p):
        captured["poses"] = p
        return []
    monkeypatch.setattr(pipe, "plan_grasps", fake_plan)
    res = pipe.run(np.zeros((4, 4, 3)), np.zeros((4, 4)), np.eye(3),
                   instruction="pick the red cube")
    assert [p.obj_id for p in captured["poses"]] == [0]      # filtró al target
    assert res.grounding is not None and res.grounding.target_obj_id == 0
    assert res.instruction is not None and res.instruction.target.color == "red"
    assert "language_grounding" in res.timing


def test_run_ignora_instruccion_si_language_disabled(monkeypatch):
    cfg = PipelineConfig(language_enabled=False)   # desactivado
    pipe = BinPickingPipeline(cfg)
    pipe._initialized = True
    poses = [_pose(0, -0.2, "red", "cube"), _pose(1, 0.2, "blue", "cube")]
    monkeypatch.setattr(pipe, "estimate_poses", lambda *a, **k: list(poses))
    captured = {}
    monkeypatch.setattr(pipe, "plan_grasps",
                        lambda p: captured.setdefault("poses", p) or [])
    res = pipe.run(np.zeros((4, 4, 3)), np.zeros((4, 4)), np.eye(3),
                   instruction="pick the red cube")
    assert [p.obj_id for p in captured["poses"]] == [0, 1]   # NO filtró (guard off)
    assert res.grounding is None and res.instruction is None
    assert "language_grounding" not in res.timing


def test_cli_language_parsea_args():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    mod = importlib.import_module("experiments.run_pick_language")
    args = mod.build_parser().parse_args(
        ["--instruction", "pick the red cube", "--scene", "multi", "--dry-run"]
    )
    assert args.instruction == "pick the red cube"
    assert args.scene == "multi"
    assert args.dry_run is True


def test_cli_dry_run_ejecuta_grounding(capsys):
    import json
    from experiments.run_pick_language import run_dry
    code = run_dry("dame el cubo rojo de la izquierda")
    out = capsys.readouterr().out
    assert code == 0
    assert '"target_obj_id"' in out
    payload = json.loads(out)
    assert payload["grounding"]["target_obj_id"] == 0
    assert payload["parsed"]["color"] == "red"
