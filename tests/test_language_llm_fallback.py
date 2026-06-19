"""El backend LLM local cae al determinista si Ollama no está disponible."""
from src.language import make_parser
from src.language.schema import Instruction


def test_llm_local_fallback_sin_ollama(monkeypatch):
    # Forzar 'no disponible'
    import src.language.backends.llm_local as m
    monkeypatch.setattr(m, "_ollama_available", lambda host: False)
    parser = make_parser("llm_local")
    instr = parser.parse("pick the red cube")
    assert isinstance(instr, Instruction)
    assert instr.target.color == "red"            # lo resolvió el fallback determinista
    assert "deterministic" in instr.backend       # backend marca el fallback


def test_llm_local_usa_respuesta_valida(monkeypatch):
    import src.language.backends.llm_local as m
    monkeypatch.setattr(m, "_ollama_available", lambda host: True)
    fake = '{"intent": "pick", "color": "blue", "shape": "sphere", "size": null, "relation": null}'
    monkeypatch.setattr(m, "_ollama_generate", lambda *a, **k: fake)
    parser = make_parser("llm_local")
    instr = parser.parse("agarra esa bolita azul de ahí")
    assert instr.target.color == "blue"
    assert instr.target.shape == "sphere"
    assert instr.backend.startswith("llm_local")


def test_llm_local_json_invalido_cae_a_determinista(monkeypatch):
    import src.language.backends.llm_local as m
    monkeypatch.setattr(m, "_ollama_available", lambda host: True)
    monkeypatch.setattr(m, "_ollama_generate", lambda *a, **k: "no soy json")
    parser = make_parser("llm_local")
    instr = parser.parse("pick the green object")
    assert instr.target.color == "green"
    assert "deterministic" in instr.backend


def test_llm_local_json_vacio_cae_a_determinista(monkeypatch):
    import src.language.backends.llm_local as m
    monkeypatch.setattr(m, "_ollama_available", lambda host: True)
    fake = '{"intent": "pick", "color": null, "shape": null, "size": null, "relation": null}'
    monkeypatch.setattr(m, "_ollama_generate", lambda *a, **k: fake)
    parser = make_parser("llm_local")
    instr = parser.parse("agarra la pieza azul")   # determinista sí ve "azul"
    assert instr.target.color == "blue"
    assert "deterministic" in instr.backend


def test_llm_local_intent_invalido_se_normaliza(monkeypatch):
    import src.language.backends.llm_local as m
    monkeypatch.setattr(m, "_ollama_available", lambda host: True)
    fake = '{"intent": "grasp", "color": "red", "shape": null, "size": null, "relation": null}'
    monkeypatch.setattr(m, "_ollama_generate", lambda *a, **k: fake)
    parser = make_parser("llm_local")
    instr = parser.parse("pick the red one")
    assert instr.target.color == "red"
    assert instr.intent == "pick"          # "grasp" no es válido -> normalizado a "pick"
    assert instr.backend.startswith("llm_local")
