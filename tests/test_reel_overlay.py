"""Tests para src/simulation/reel_overlay.py."""
import numpy as np

from src.simulation.reel_overlay import (
    draw_honesty_tag,
    draw_metrics,
    draw_title_bar,
    make_title_card,
    normalize_frame,
    truncate,
)


def test_normalize_frame_640x480_to_720p():
    src = np.full((480, 640, 3), 128, dtype=np.uint8)
    out = normalize_frame(src)
    assert out.shape == (720, 1280, 3)
    assert out.dtype == np.uint8
    # Letterbox: filas superiores/inferiores deben quedar negras (pad)
    assert out[0, 0].sum() == 0


def test_normalize_frame_already_720p_unchanged_shape():
    src = np.full((720, 1280, 3), 200, dtype=np.uint8)
    out = normalize_frame(src)
    assert out.shape == (720, 1280, 3)


def test_truncate_adds_ellipsis():
    assert truncate("abcdefghij", 5) == "abcd…"
    assert truncate("abc", 5) == "abc"


def test_draw_title_bar_modifies_top_band_only():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = draw_title_bar(frame, "①", "FoundationPose")
    # La banda superior (y<70) debe tener pixels no nulos
    assert out[20, 100].sum() > 0
    # El centro de la imagen no debe haberse tocado
    assert out[400, 640].sum() == 0


def test_draw_metrics_green_when_ok():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = draw_metrics(frame, [("IK convergido ✓", True)])
    # Región lower-left debe tener pixels dibujados
    assert out[620:700, 16:600].sum() > 0


def test_draw_honesty_tag_modifies_bottom_right():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = draw_honesty_tag(frame, "grasp por attach")
    assert out[690:715, 800:1280].sum() > 0


def test_make_title_card_shape_and_text():
    card = make_title_card([("Bin-picking 6-DoF", 1.2), ("subtitulo", 0.7)])
    assert card.shape == (720, 1280, 3)
    # Centro vertical debe tener texto (pixels no nulos)
    assert card[300:420, :].sum() > 0
