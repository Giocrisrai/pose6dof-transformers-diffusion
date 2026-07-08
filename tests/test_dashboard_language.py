"""Smoke test del tab de PLN del dashboard (lógica pura, sin runtime)."""
import pytest

pytest.importorskip("streamlit")


def test_render_language_tab_no_revienta():
    from dashboard import render_language_tab  # función pura, sin st.run completo
    payload = render_language_tab("dame el cubo rojo de la izquierda")
    assert payload["grounding"]["target_obj_id"] is not None
    assert payload["parsed"]["color"] == "red"


def test_render_language_tab_sin_match():
    from dashboard import render_language_tab
    payload = render_language_tab("pick the green cylinder")
    assert payload["grounding"]["target_obj_id"] is None


def test_dashboard_text_to_cad_section_renders():
    """La sección text-to-CAD del dashboard carga y muestra sus métricas."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("dashboard.py", default_timeout=60)
    at.run()
    assert not at.exception
    assert any("text-to-CAD" in o for o in at.radio[0].options)
    at.radio[0].set_value("🛠️ text-to-CAD").run()
    assert not at.exception
    assert any("text-to-CAD" in t.value for t in at.title)
    assert len(at.metric) >= 7
