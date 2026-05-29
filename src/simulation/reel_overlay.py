"""Overlays para el demo reel (cv2, sin ffmpeg drawtext).

El ffmpeg local no tiene libfreetype/drawtext, así que todo el texto se
renderiza con cv2.putText (fuentes Hershey) sobre los frames. Reutiliza el
estilo de experiments/record_e2e_video_v2.py.

Convención de color: BGR (como cv2).
"""
from __future__ import annotations

import cv2
import numpy as np

W, H = 1280, 720
_FONT_TITLE = cv2.FONT_HERSHEY_DUPLEX
_FONT_BODY = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_MUTED = (150, 150, 150)
_GREEN = (90, 220, 120)   # BGR
_ACCENT = (205, 152, 10)  # ámbar BGR


def normalize_frame(frame: np.ndarray, w: int = W, h: int = H) -> np.ndarray:
    """Escala manteniendo aspecto y pad (letterbox) a (h, w, 3)."""
    src_h, src_w = frame.shape[:2]
    scale = min(w / src_w, h / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def truncate(text: str, max_chars: int) -> str:
    """Trunca con elipsis si excede max_chars."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _alpha_band(frame, x0, y0, x1, y1, color, alpha):
    """Blend de un rectángulo relleno semi-transparente, in-place."""
    sub = frame[y0:y1, x0:x1]
    overlay = np.full_like(sub, color, dtype=np.uint8)
    frame[y0:y1, x0:x1] = cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0)


def draw_title_bar(frame: np.ndarray, number: str, title: str,
                   accent=_ACCENT) -> np.ndarray:
    """Banda superior semi-transparente + franja de acento + título."""
    _alpha_band(frame, 0, 0, W, 70, (30, 30, 38), 0.55)
    cv2.rectangle(frame, (0, 0), (8, 70), accent, -1)
    label = f"{number} {truncate(title, 60)}".strip()
    cv2.putText(frame, label, (24, 46), _FONT_TITLE, 0.95, _WHITE, 2, cv2.LINE_AA)
    return frame


def draw_metrics(frame: np.ndarray, lines, accent=_ACCENT) -> np.ndarray:
    """Panel lower-left con métricas. lines = list[(texto, pasa_threshold)].

    Si pasa_threshold es True, el texto va en verde; si no, en blanco.
    """
    n = len(lines)
    panel_h = 18 + 30 * n
    y_top = H - 20 - panel_h
    _alpha_band(frame, 0, y_top, 620, H - 8, (30, 30, 38), 0.55)
    cv2.rectangle(frame, (0, y_top), (8, H - 8), accent, -1)
    y = y_top + 34
    for text, ok in lines:
        color = _GREEN if ok else _WHITE
        cv2.putText(frame, truncate(text, 52), (24, y), _FONT_BODY, 0.62,
                    color, 1, cv2.LINE_AA)
        y += 30
    return frame


def draw_honesty_tag(frame: np.ndarray, text: str) -> np.ndarray:
    """Texto chico y tenue abajo-derecha."""
    label = truncate(text, 60)
    (tw, _), _ = cv2.getTextSize(label, _FONT_BODY, 0.5, 1)
    x = W - tw - 20
    cv2.putText(frame, label, (x, H - 18), _FONT_BODY, 0.5, _MUTED, 1, cv2.LINE_AA)
    return frame


def make_title_card(lines, w: int = W, h: int = H, accent=_ACCENT) -> np.ndarray:
    """Frame oscuro con líneas centradas. lines = list[(texto, escala)]."""
    card = np.full((h, w, 3), (28, 28, 34), dtype=np.uint8)
    cv2.rectangle(card, (0, h // 2 - 90), (10, h // 2 + 90), accent, -1)
    total = sum(int(40 * s) + 18 for _, s in lines)
    y = h // 2 - total // 2 + 30
    for text, scale in lines:
        (tw, th), _ = cv2.getTextSize(text, _FONT_TITLE, scale, 2)
        x = (w - tw) // 2
        cv2.putText(card, text, (x, y), _FONT_TITLE, scale, _WHITE, 2, cv2.LINE_AA)
        y += int(40 * scale) + 18
    return card
