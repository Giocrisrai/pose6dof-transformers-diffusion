"""Smoke test del tab de PLN del dashboard (lógica pura, sin runtime)."""
import pytest

pytest.importorskip("streamlit")


def test_render_language_tab_no_revienta():
    from dashboard import render_language_tab   # función pura, sin st.run completo
    payload = render_language_tab("dame el cubo rojo de la izquierda")
    assert payload["grounding"]["target_obj_id"] is not None
    assert payload["parsed"]["color"] == "red"


def test_render_language_tab_sin_match():
    from dashboard import render_language_tab
    payload = render_language_tab("pick the green cylinder")
    assert payload["grounding"]["target_obj_id"] is None
