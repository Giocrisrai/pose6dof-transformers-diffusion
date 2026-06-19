"""Tests del parser determinista y del léxico."""
import pytest

from src.language import make_parser, vocab
from src.language.schema import Instruction


def test_normaliza_color_es_en():
    assert vocab.normalize_color("rojo") == "red"
    assert vocab.normalize_color("red") == "red"
    assert vocab.normalize_color("AZUL") == "blue"
    assert vocab.normalize_color("morado") is None  # fuera de vocabulario


def test_normaliza_forma():
    assert vocab.normalize_shape("cubo") == "cube"
    assert vocab.normalize_shape("esfera") == "sphere"
    assert vocab.normalize_shape("cylinder") == "cylinder"


def test_normaliza_size_y_relacion():
    assert vocab.normalize_size("pequeño") == "small"
    assert vocab.normalize_size("grande") == "large"
    assert vocab.normalize_relation("a la izquierda") == "left_of"
    assert vocab.normalize_relation("más cercano") == "nearest"


def test_no_falsos_positivos_por_substring():
    assert vocab.normalize_relation("laptop on the desk") != "on_top" or vocab.normalize_relation("laptop") is None
    assert vocab.normalize_relation("laptop") is None
    assert vocab.normalize_shape("xbox") is None
    assert vocab.normalize_shape("balloon") is None
    assert vocab.normalize_relation("leftover parts") is None


def test_frase_vs_palabra_longest_match():
    assert vocab.normalize_shape("la caja cuadrada") == "cube"
    assert vocab.normalize_shape("la caja") == "box"


def test_normaliza_resto_valores_canonicos():
    assert vocab.normalize_size("big") == "large"
    assert vocab.normalize_relation("a la derecha") == "right_of"
    assert vocab.normalize_relation("más lejano") == "farthest"
    assert vocab.normalize_relation("encima") == "on_top"


def test_parser_extrae_color_y_forma():
    p = make_parser("deterministic")
    instr = p.parse("pick the red cube")
    assert isinstance(instr, Instruction)
    assert instr.target.color == "red"
    assert instr.target.shape == "cube"
    assert instr.intent == "pick"
    assert instr.backend == "deterministic"


def test_parser_espanol_con_relacion_espacial():
    p = make_parser("deterministic")
    instr = p.parse("dame el cubo rojo de la izquierda")
    assert instr.target.color == "red"
    assert instr.target.shape == "cube"
    assert instr.spatial is not None
    assert instr.spatial.relation == "left_of"


def test_parser_tamano_y_sustantivo():
    instr = make_parser("deterministic").parse("agarra la pieza pequeña azul")
    assert instr.target.size == "small"
    assert instr.target.color == "blue"
    assert instr.target.raw_noun == "pieza"


def test_parser_secuencia_dos_pasos():
    instr = make_parser("deterministic").parse(
        "pick the red cube and then the blue sphere"
    )
    assert instr.intent == "sequence"
    assert len(instr.steps) == 2
    assert instr.steps[0].target.color == "red"
    assert instr.steps[1].target.shape == "sphere"


def test_parser_frase_vacia_no_rompe():
    instr = make_parser("deterministic").parse("")
    assert instr.target.is_empty()
    assert instr.confidence < 1.0


def test_make_parser_desconocido_lanza():
    with pytest.raises(ValueError):
        make_parser("inexistente")
