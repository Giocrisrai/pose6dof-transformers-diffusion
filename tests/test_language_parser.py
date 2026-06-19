"""Tests del parser determinista y del léxico."""
from src.language import vocab


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
