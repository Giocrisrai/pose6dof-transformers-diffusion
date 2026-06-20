#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Genera el .docx de la Entrega 4 del TFM.

Parte de una copia de la Entrega 3 e inserta, en tres puntos de anclaje, la
contribucion de la Entrega 4: una capa de bin picking guiado por lenguaje
natural (arquitectura, ejecucion E2E, resultados, tablas, figura y limitaciones),
una comparacion con el estado del arte y un cierre en conclusiones.

El script es idempotente: siempre re-copia desde la Entrega 3 al iniciar, de modo
que volver a ejecutarlo regenera limpiamente el documento sin duplicar contenido.

IMPORTANTE: requiere python-docx, que SOLO esta instalado en `../.venv_thesis`.
Ejecutar con:
    ../.venv_thesis/bin/python scripts/build_entrega4_docx.py
"""

import os
import shutil
import sys

from docx import Document
from docx.shared import Inches, Pt  # noqa: F401  (Pt disponible por si se ajustan tamanos)
from docx.enum.text import WD_ALIGN_PARAGRAPH

# --- Rutas (relativas a la raiz del repo repo_tfm) ---------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E3_PATH = os.path.join(REPO_ROOT, "docs", "entrega3", "TFM_Entrega3_UNIR.docx")
E4_DIR = os.path.join(REPO_ROOT, "docs", "entrega4")
E4_PATH = os.path.join(E4_DIR, "TFM_Entrega4_UNIR.docx")
FIG_PATH = os.path.join(E4_DIR, "assets", "fig_language_pick.png")


# --- Helpers -----------------------------------------------------------------
def find_anchor(doc, style_name, text):
    """Devuelve el primer parrafo con el estilo y texto dados, o falla ruidosamente.

    El emparejamiento es por `style.name` exacto y por `text.strip()` exacto, de
    modo que un cambio futuro en el documento haga fallar el script de forma clara
    en lugar de insertar contenido en un lugar incorrecto.
    """
    for p in doc.paragraphs:
        if p.style is not None and p.style.name == style_name and p.text.strip() == text:
            return p
    raise RuntimeError(
        "No se encontro el ancla: estilo=%r texto=%r. "
        "El documento base ha cambiado; revise los anclajes." % (style_name, text)
    )


def insert_para(anchor, text, style=None):
    """Inserta un parrafo (con estilo opcional) inmediatamente antes del ancla."""
    return anchor.insert_paragraph_before(text, style=style)


def insert_caption(anchor, text):
    """Inserta un parrafo de leyenda en cursiva antes del ancla.

    Usa el estilo 'Caption' si existe; en caso contrario, un parrafo normal con
    la corrida en cursiva.
    """
    try:
        cap = anchor.insert_paragraph_before(text, style="Caption")
        # Asegura cursiva tambien sobre el estilo Caption.
        for run in cap.runs:
            run.italic = True
        if not cap.runs:
            cap.add_run(text).italic = True
        return cap
    except (KeyError, ValueError):
        cap = anchor.insert_paragraph_before("")
        cap.add_run(text).italic = True
        return cap


def insert_table_before(doc, anchor, header, rows, table_style):
    """Crea una tabla (cabecera + filas) y la mueve a la posicion previa al ancla."""
    n_cols = len(header)
    t = doc.add_table(rows=1 + len(rows), cols=n_cols)
    if table_style is not None:
        t.style = table_style
    # Cabecera
    for c, txt in enumerate(header):
        cell = t.cell(0, c)
        cell.text = txt
        # Cabecera en negrita.
        for run in cell.paragraphs[0].runs:
            run.bold = True
    # Filas de datos
    for r, row in enumerate(rows, start=1):
        for c, txt in enumerate(row):
            t.cell(r, c).text = txt
    # Mover la tabla (creada al final del documento) a su posicion correcta.
    anchor._p.addprevious(t._tbl)
    return t


def insert_figure(anchor, image_path, width):
    """Inserta una imagen centrada inmediatamente antes del ancla."""
    p = anchor.insert_paragraph_before("")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(image_path, width=width)
    return p


# --- Contenido a insertar (verbatim) -----------------------------------------
A4_ARCH_TITLE = "Bin picking guiado por lenguaje natural: arquitectura"
A4_ARCH_BODY = (
    "La contribución de la Entrega 4 añade una capa de comprensión de lenguaje "
    "natural sobre el pipeline de bin picking descrito en las secciones "
    "anteriores, de modo que el sistema acepta una instrucción —por ejemplo, "
    "«dame el cubo rojo de la izquierda»— y selecciona y manipula el objeto "
    "descrito. La capa se organiza en dos unidades desacopladas. La primera, el "
    "analizador de instrucciones, traduce el texto a una representación "
    "estructurada (intención, atributos del objetivo —color, forma y tamaño— y "
    "relación espacial); su implementación por defecto es determinista, basada "
    "en un léxico controlado español-inglés con coincidencia por límites de "
    "palabra, y admite, de forma intercambiable, un backend con un modelo de "
    "lenguaje local y un backend de API, ambos con degradación elegante al "
    "analizador determinista. La segunda unidad, el anclaje, asocia la "
    "instrucción a los objetos detectados puntuando la coincidencia de atributos "
    "y resolviendo la ambigüedad mediante la relación espacial sobre los "
    "centroides. La integración en el pipeline es opcional y no intrusiva: se "
    "activa mediante un parámetro de configuración y, cuando se proporciona una "
    "instrucción, filtra los objetos candidatos al objetivo descrito antes de la "
    "planificación del agarre."
)
A4_E2E_TITLE = "Ejecución extremo a extremo en simulación"
A4_E2E_BODY = (
    "Para validar la capa en condiciones de ejecución se implementó una rutina "
    "extremo a extremo en CoppeliaSim. Dicha rutina construye una escena "
    "multiobjeto con variedad de formas reales —cubos, esferas y cilindros "
    "generados como primitivas—, ancla la instrucción al objeto correspondiente "
    "y ejecuta la secuencia completa de pick-and-place sobre el objeto "
    "seleccionado, reutilizando el controlador de cinemática inversa y la técnica "
    "de agarre del pipeline base. La métrica de selección empleada es honesta: "
    "una selección se considera correcta únicamente si el objeto manipulado "
    "coincide con todos los atributos especificados en la instrucción, lo que "
    "evita falsos aciertos por coincidencias parciales."
)
A4_RES_TITLE = "Resultados y validación de la capa de lenguaje"
A4_RES_BODY = (
    "La capa de lenguaje se validó tanto de forma reproducible sin simulador "
    "como en ejecución real sobre CoppeliaSim. La Tabla de capacidades resume "
    "las funcionalidades añadidas respecto a la Entrega 3 y la Tabla de "
    "validación recoge la evidencia obtenida. En la batería de selección "
    "reproducible, con noventa escenas y variedad de formas, la precisión de "
    "selección fue del 100 % en las tres familias evaluadas —color, forma y "
    "relación espacial—; conviene precisar que este resultado valida la cadena "
    "completa de análisis, anclaje y selección sobre un banco controlado en el "
    "que el objeto descrito es unívoco, mientras que la robustez frente a "
    "lenguaje ambiguo o no visto queda evidenciada por las exploraciones exp16 a "
    "exp26 sobre el modelo aprendido. En ejecución real, el test de integración "
    "completó el ciclo en noventa y ocho segundos; una ejecución representativa "
    "de la instrucción «dame el cubo rojo de la izquierda» seleccionó "
    "correctamente el objetivo con una proximidad pinza-objeto de cuatro "
    "milímetros, por debajo del umbral de plausibilidad de cinco centímetros, y "
    "con convergencia de la cinemática inversa; y la batería en simulador, con "
    "nueve escenas, alcanzó el 100 % de selección correcta."
)

CAP_HEADER = ["Capacidad", "Entrega 3", "Entrega 4"]
CAP_ROWS = [
    ["Pose 6-DoF, agarre y servoing", "Sí (H1/H2/H3 validadas)", "Sí (preservado)"],
    ["Selección por lenguaje natural", "No", "Sí"],
    ["Anclaje por atributos y relación espacial", "No", "Sí"],
    ["Variedad de formas en simulación", "Cubos", "Cubo, esfera y cilindro"],
    ["Ejecución E2E por instrucción", "No", "Sí"],
    ["Métrica de selección honesta", "—", "Sí"],
]
CAP_CAPTION = "Tabla. Comparación de capacidades entre la Entrega 3 y la Entrega 4."

VAL_HEADER = ["Prueba", "Resultado", "Fuente"]
VAL_ROWS = [
    ["Batería de selección (pura, 90 escenas)", "100 % (color, forma y espacial)", "report_pure_n90.json"],
    ["Test de integración E2E", "Completado en 98 s", "test de integración"],
    ["Ejecución E2E representativa", "Selección correcta; agarre a 4 mm; IK convergente", "run en vivo"],
    ["Batería de selección en simulador (9 escenas)", "100 %", "report_sim_n9.json"],
    ["Reel de demostración", "174 s, 5 instrucciones", "language_reel.mp4"],
]
VAL_CAPTION = "Tabla. Evidencia de validación de la capa de lenguaje (junio de 2026)."

FIG_CAPTION = (
    "Figura. Ejecución extremo a extremo de la instrucción «dame el cubo rojo de "
    "la izquierda» en una escena con variedad de formas."
)

A4_LIM_TITLE = "Limitaciones y trabajo futuro de la capa de lenguaje"
A4_LIM_BODY = (
    "La capa de lenguaje presenta limitaciones que conviene explicitar. El "
    "agarre es cinemático, por lo que la métrica honesta de calidad es la "
    "proximidad pinza-objeto y la convergencia de la cinemática inversa, y no el "
    "desplazamiento del objeto. El banco de selección descrito es controlado y "
    "valida la cadena de ejecución, pero no discrimina casos genuinamente "
    "ambiguos —dos objetos del mismo color y forma—, escenario que se propone "
    "como trabajo futuro. La estimación de tamaño mediante el modelo de "
    "visión-lenguaje es el atributo menos fiable al carecer de una referencia de "
    "escala. Por último, el backend de API es un punto de extensión sin proveedor "
    "concreto integrado."
)

A2_BODY = (
    "En relación con los trabajos de manipulación guiada por lenguaje del estado "
    "del arte —CLIPort, VoxPoser, SayCan u OWL-ViT y CLIP-Fields—, la "
    "contribución propuesta en este trabajo se posiciona por su carácter "
    "íntegramente open-license y su ejecución en un equipo portátil sin GPU "
    "dedicada, junto con un anclaje interpretable y una comprensión basada en "
    "modelos de lenguaje intercambiables que no exige reentrenamiento por "
    "vocabulario. No se reporta una cifra única de precisión comparable con "
    "dichos trabajos, por evaluarse sobre bancos heterogéneos, por lo que la "
    "comparación se mantiene cualitativa en esos ejes."
)

A5_BODY = (
    "La Entrega 4 incorpora, sobre el pipeline validado, una capa de bin picking "
    "guiado por lenguaje natural que permite seleccionar y manipular el objeto "
    "descrito por una instrucción, validada extremo a extremo en simulación. "
    "Como trabajo futuro se plantea un banco de selección con casos genuinamente "
    "ambiguos, el anclaje por imagen sobre la cámara real del simulador y la "
    "integración de un proveedor concreto en el backend de API."
)

NEW_H3_TITLES = [A4_ARCH_TITLE, A4_E2E_TITLE, A4_RES_TITLE, A4_LIM_TITLE]


# --- Construccion ------------------------------------------------------------
def build():
    if not os.path.exists(E3_PATH):
        raise FileNotFoundError("No existe la Entrega 3: %s" % E3_PATH)
    if not os.path.exists(FIG_PATH):
        raise FileNotFoundError("No existe la figura: %s" % FIG_PATH)
    os.makedirs(E4_DIR, exist_ok=True)

    # Idempotencia: re-copiar siempre desde E3.
    shutil.copy(E3_PATH, E4_PATH)

    doc = Document(E4_PATH)

    # Estilo de tabla reutilizado del propio documento.
    table_style = doc.tables[0].style if doc.tables else None

    # ---- Antes de A4 (Capitulo 4), en orden de lectura ----
    a4 = find_anchor(doc, "Heading 3", "Resumen consolidado de la evidencia experimental")
    insert_para(a4, A4_ARCH_TITLE, style="Heading 3")
    insert_para(a4, A4_ARCH_BODY, style="Normal")
    insert_para(a4, A4_E2E_TITLE, style="Heading 3")
    insert_para(a4, A4_E2E_BODY, style="Normal")
    insert_para(a4, A4_RES_TITLE, style="Heading 3")
    insert_para(a4, A4_RES_BODY, style="Normal")
    insert_table_before(doc, a4, CAP_HEADER, CAP_ROWS, table_style)
    insert_caption(a4, CAP_CAPTION)
    insert_table_before(doc, a4, VAL_HEADER, VAL_ROWS, table_style)
    insert_caption(a4, VAL_CAPTION)
    insert_figure(a4, FIG_PATH, Inches(5.5))
    insert_caption(a4, FIG_CAPTION)
    insert_para(a4, A4_LIM_TITLE, style="Heading 3")
    insert_para(a4, A4_LIM_BODY, style="Normal")

    # ---- Antes de A2 (Capitulo 2, estado del arte) ----
    a2 = find_anchor(doc, "Heading 2", "Conclusiones del estado del arte y brecha de investigación")
    insert_para(a2, A2_BODY, style="Normal")

    # ---- Antes de A5 (Capitulo 5, conclusiones) ----
    a5 = find_anchor(doc, "Heading 2", "Discusión y cierre de hipótesis")
    insert_para(a5, A5_BODY, style="Normal")

    doc.save(E4_PATH)
    return doc


# --- Verificacion ------------------------------------------------------------
def heading_texts(doc):
    out = []
    for p in doc.paragraphs:
        if p.style is not None and p.style.name in ("Heading 1", "Heading 2", "Heading 3"):
            out.append(p.text.strip())
    return out


def verify():
    e3 = Document(E3_PATH)
    e3_tables = len(e3.tables)
    e3_shapes = len(e3.inline_shapes)
    e3_headings = set(t for t in heading_texts(e3) if t)

    e4 = Document(E4_PATH)  # se reabre: si no abre, lanza excepcion
    e4_tables = len(e4.tables)
    e4_shapes = len(e4.inline_shapes)
    e4_heading_list = heading_texts(e4)
    e4_headings = set(t for t in e4_heading_list if t)

    print("== VERIFICACION ENTREGA 4 ==")
    print("Documento E4 abierto correctamente: OK")

    # 4 nuevos titulos Heading 3
    for title in NEW_H3_TITLES:
        present = title in e4_heading_list
        print("Heading 3 presente [%s]: %s" % ("OK" if present else "FALTA", title))
        assert present, "Falta el Heading 3: %s" % title

    # Tablas: E4 == E3 + 2
    print("Tablas E3=%d  E4=%d  (esperado E3+2=%d)" % (e3_tables, e4_tables, e3_tables + 2))
    assert e4_tables == e3_tables + 2, "Conteo de tablas inesperado"

    # Imagenes: E4 == E3 + 1
    print("Imagenes (inline_shapes) E3=%d  E4=%d  (esperado E3+1=%d)" % (e3_shapes, e4_shapes, e3_shapes + 1))
    assert e4_shapes == e3_shapes + 1, "Conteo de imagenes inesperado"

    # Parrafos clave SOTA y conclusiones
    full_text = "\n".join(p.text for p in e4.paragraphs)
    sota_ok = "íntegramente open-license" in full_text
    concl_ok = "validada extremo a extremo en simulación" in full_text
    print("Parrafo estado del arte ('íntegramente open-license'): %s" % ("OK" if sota_ok else "FALTA"))
    print("Parrafo conclusiones ('validada extremo a extremo en simulación'): %s" % ("OK" if concl_ok else "FALTA"))
    assert sota_ok, "Falta el parrafo del estado del arte"
    assert concl_ok, "Falta el parrafo de conclusiones"

    # Preservacion: todo heading de E3 sigue en E4
    missing = sorted(e3_headings - e4_headings)
    if missing:
        print("PRESERVACION E3: FALTAN %d encabezados:" % len(missing))
        for m in missing:
            print("   - %s" % m)
        raise AssertionError("No se preservaron todos los encabezados de E3")
    else:
        print("PRESERVACION E3: OK")

    print("== TODAS LAS VERIFICACIONES PASARON ==")
    return {
        "e3_tables": e3_tables,
        "e4_tables": e4_tables,
        "e3_shapes": e3_shapes,
        "e4_shapes": e4_shapes,
    }


def main():
    print("Generando Entrega 4 a partir de la Entrega 3...")
    print("  E3: %s" % E3_PATH)
    print("  E4: %s" % E4_PATH)
    build()
    print("Documento generado y guardado.")
    verify()
    print("Listo: %s" % E4_PATH)


if __name__ == "__main__":
    main()
    sys.exit(0)
