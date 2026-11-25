"""La escena base no debe cambiar sin reentrenar las políticas visuales.

Las políticas v8 y v9 reciben la imagen de la cámara, donde se ve el brazo.
El 29 de junio de 2026 la escena se volvió a guardar con el brazo en otra
postura y el agarre cayó del 100 % a cerca del 58 % sin que ningún test lo
notara (exploración 15). Si la escena cambia a propósito, hay que volver a
recoger datos, reentrenar y actualizar esta huella.
"""
import hashlib
from pathlib import Path

ESCENA = Path(__file__).resolve().parents[1] / "data/scenes/bin_base.ttt"
HUELLA_ENTRENAMIENTO = "26ee8415dd604dea6ba9d95b88656b8747773123908242e6f5c1fce792425465"


def test_escena_base_es_la_del_entrenamiento():
    huella = hashlib.sha256(ESCENA.read_bytes()).hexdigest()
    assert huella == HUELLA_ENTRENAMIENTO, (
        "data/scenes/bin_base.ttt cambió: las políticas visuales se entrenaron con otra "
        "escena. Restaura el archivo o reentrena y actualiza la huella."
    )
