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
