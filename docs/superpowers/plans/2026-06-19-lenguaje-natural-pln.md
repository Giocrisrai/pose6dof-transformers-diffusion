# Bin picking guiado por lenguaje natural (PLN) — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Añadir al pipeline una capacidad open-source de bin picking guiado por lenguaje natural ("dame el cubo rojo de la izquierda") integrada, testeada y documentada.

**Architecture:** Paquete `src/language/` con dos unidades desacopladas — `InstructionParser` (texto→`Instruction` estructurada, default determinista, LLM local enchufable) y `Grounder` (`Instruction` + objetos detectados→target). Se integra en `BinPickingPipeline.run(instruction=...)` a nivel de selección de target, reusando el grounding ya validado (exp16–26). Default 100% reproducible sin red ni pesos; CLIP y LLM son opcionales con fallback.

**Tech Stack:** Python 3.12, numpy, pytest; opcional `transformers`/CLIP (grounding por imagen), `ollama`/`mlx-lm` (parser LLM, Fase 2); Streamlit (dashboard).

**Convenciones del repo:**
- Tests en `tests/test_*.py`, `pythonpath=["."]`, import como `from src.language... import ...`.
- Comentarios y docstrings en español (proyecto personal).
- Marcador nuevo en `pyproject.toml`: `slow` para tests que requieren CLIP/LLM.
- Commits frecuentes; rama actual `feat/lenguaje-natural-pln`.

---

## Estructura de ficheros

```
src/language/
  __init__.py            # exporta API pública: Instruction, make_parser, Grounder, ...
  schema.py              # dataclasses puras (sin deps pesadas)
  vocab.py               # léxico controlado ES/EN + normalización
  parser.py              # interfaz InstructionParser + make_parser factory
  grounding.py           # Grounder (determinista + CLIP opcional)
  backends/
    __init__.py
    deterministic.py     # DeterministicParser (Fase 1, default)
    llm_local.py         # LLMLocalParser (Fase 2, Ollama/MLX)
    llm_api.py           # LLMApiParser (Fase 2, stub configurable)

experiments/run_pick_language.py   # CLI de demo (nuevo)
tests/test_language_schema.py
tests/test_language_parser.py
tests/test_language_grounding.py
tests/test_pipeline_language.py
tests/test_language_llm_fallback.py

docs/LENGUAJE_NATURAL.md
docs/COMPARATIVA_SOTA_LENGUAJE.md
```

Modificados: `src/pipeline.py`, `dashboard.py`, `pyproject.toml`, `README.md`,
`docs/exploraciones/` (nota de consolidación), memoria del proyecto.

---

## FASE 1 — Núcleo determinista integrado

### Task 1: Schema de datos (`src/language/schema.py`)

**Files:**
- Create: `src/language/__init__.py`
- Create: `src/language/schema.py`
- Test: `tests/test_language_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_language_schema.py
"""Tests para src/language/schema.py — dataclasses puras."""
from src.language.schema import (
    TargetSpec, SpatialRelation, Instruction, GroundingResult, ObjectView,
)


def test_targetspec_defaults_son_none():
    t = TargetSpec()
    assert t.color is None and t.shape is None and t.size is None
    assert t.raw_noun is None
    assert t.is_empty()


def test_targetspec_no_vacio_con_atributo():
    assert not TargetSpec(color="red").is_empty()


def test_instruction_minima():
    instr = Instruction(raw_text="pick the red cube", target=TargetSpec(color="red", shape="cube"))
    assert instr.intent == "pick"          # default
    assert instr.steps == []               # default
    assert instr.spatial is None
    assert instr.confidence == 1.0
    assert instr.backend == "unknown"


def test_objectview_y_grounding_result():
    ov = ObjectView(obj_id=2, centroid=(0.1, 0.2, 0.3), attributes={"color": "red"})
    assert ov.obj_id == 2
    g = GroundingResult(target_obj_id=2, scores={2: 0.9, 1: 0.1},
                        method="attribute", rejected=[1], ambiguous=False)
    assert g.target_obj_id == 2 and not g.ambiguous
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_language_schema.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.language'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/language/__init__.py
"""Capacidad de bin picking guiado por lenguaje natural (PLN).

API pública estable:
    from src.language import make_parser, Grounder, Instruction
"""
from src.language.schema import (  # noqa: F401
    TargetSpec, SpatialRelation, Instruction, GroundingResult, ObjectView,
)
```

```python
# src/language/schema.py
"""Modelo de datos del subsistema de lenguaje natural.

Dataclasses puras, sin dependencias pesadas (numpy/torch/CLIP). Son el
contrato entre el parser (texto -> Instruction) y el grounder
(Instruction + objetos -> target).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TargetSpec:
    """Atributos que describen el objeto buscado."""
    color: Optional[str] = None        # normalizado: "red", "blue", "green", ...
    shape: Optional[str] = None        # "cube", "sphere", "cylinder", "box"
    size: Optional[str] = None         # "small", "large" (atributo continuo, exp22)
    raw_noun: Optional[str] = None     # sustantivo crudo: "pieza", "objeto"

    def is_empty(self) -> bool:
        """True si no hay ningún atributo discriminativo."""
        return not (self.color or self.shape or self.size)


@dataclass
class SpatialRelation:
    """Relación espacial respecto a un ancla o a la escena (exp26)."""
    relation: str                          # left_of/right_of/nearest/farthest/on_top
    anchor: Optional[TargetSpec] = None    # objeto de referencia (si aplica)


@dataclass
class Instruction:
    """Instrucción de lenguaje natural ya estructurada."""
    raw_text: str
    target: TargetSpec
    intent: str = "pick"                   # pick | pick_then_place | sequence
    spatial: Optional[SpatialRelation] = None
    steps: list["Instruction"] = field(default_factory=list)  # exp23
    confidence: float = 1.0
    backend: str = "unknown"               # qué parser lo produjo


@dataclass
class ObjectView:
    """Vista ligera de un objeto detectado, consumible por el Grounder.

    Desacopla el grounder de PoseResult: el pipeline construye ObjectViews
    a partir de sus PoseResult (+ atributos de CLIP o de metadatos de escena).
    """
    obj_id: int
    centroid: tuple[float, float, float]   # posición 3D en metros (cámara/mundo)
    attributes: dict = field(default_factory=dict)  # {"color","shape","size"}
    bbox: Optional[tuple] = None           # (x1,y1,x2,y2) en imagen, para crops CLIP


@dataclass
class GroundingResult:
    """Resultado de asociar una Instruction a los objetos de la escena."""
    target_obj_id: Optional[int]
    scores: dict                           # {obj_id: score}
    method: str                            # "attribute" | "clip_image" | "spatial"
    rejected: list = field(default_factory=list)
    ambiguous: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_language_schema.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/language/__init__.py src/language/schema.py tests/test_language_schema.py
git commit -m "feat(language): schema de datos del subsistema PLN"
```

---

### Task 2: Léxico controlado (`src/language/vocab.py`)

**Files:**
- Create: `src/language/vocab.py`
- Test: `tests/test_language_parser.py` (sección vocab)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_language_parser.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_language_parser.py -q`
Expected: FAIL con `ModuleNotFoundError` / `AttributeError` (vocab no existe).

- [ ] **Step 3: Write minimal implementation**

```python
# src/language/vocab.py
"""Léxico controlado ES/EN y normalizadores.

Mapea sinónimos en español e inglés a un valor canónico en inglés
(coherente con los atributos usados en exp16-26).
"""
from __future__ import annotations

from typing import Optional

COLORS = {
    "red": ["red", "rojo", "roja"],
    "blue": ["blue", "azul"],
    "green": ["green", "verde"],
    "yellow": ["yellow", "amarillo", "amarilla"],
}
SHAPES = {
    "cube": ["cube", "cubo", "caja cuadrada"],
    "sphere": ["sphere", "esfera", "bola", "ball"],
    "cylinder": ["cylinder", "cilindro"],
    "box": ["box", "caja"],
}
SIZES = {
    "small": ["small", "pequeño", "pequeña", "chico", "little"],
    "large": ["large", "big", "grande", "gran"],
}
RELATIONS = {
    "left_of": ["left", "izquierda", "a la izquierda", "left of"],
    "right_of": ["right", "derecha", "a la derecha", "right of"],
    "nearest": ["nearest", "closest", "más cercano", "mas cercano", "cercano"],
    "farthest": ["farthest", "furthest", "más lejano", "mas lejano", "lejano"],
    "on_top": ["on top", "encima", "arriba", "top"],
}
NOUNS = ["object", "objeto", "piece", "pieza", "block", "bloque", "item"]


def _match(text: str, table: dict) -> Optional[str]:
    t = text.lower()
    # buscar la coincidencia más larga primero (frases > palabras)
    best = None
    best_len = 0
    for canonical, synonyms in table.items():
        for syn in synonyms:
            if syn in t and len(syn) > best_len:
                best, best_len = canonical, len(syn)
    return best


def normalize_color(text: str) -> Optional[str]:
    return _match(text, COLORS)


def normalize_shape(text: str) -> Optional[str]:
    return _match(text, SHAPES)


def normalize_size(text: str) -> Optional[str]:
    return _match(text, SIZES)


def normalize_relation(text: str) -> Optional[str]:
    return _match(text, RELATIONS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_language_parser.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/language/vocab.py tests/test_language_parser.py
git commit -m "feat(language): léxico controlado ES/EN y normalizadores"
```

---

### Task 3: Parser determinista + factory (`parser.py`, `backends/deterministic.py`)

**Files:**
- Create: `src/language/parser.py`
- Create: `src/language/backends/__init__.py`
- Create: `src/language/backends/deterministic.py`
- Test: `tests/test_language_parser.py` (añadir casos)

- [ ] **Step 1: Write the failing test (añadir al final del fichero)**

```python
# tests/test_language_parser.py  (añadir)
from src.language import make_parser
from src.language.schema import Instruction


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
    assert instr.target.raw_noun in ("pieza", "piece")


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
    import pytest
    with pytest.raises(ValueError):
        make_parser("inexistente")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_language_parser.py -q`
Expected: FAIL con `ImportError: cannot import name 'make_parser'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/language/parser.py
"""Interfaz InstructionParser y factory make_parser.

El default es el parser determinista (sin red ni pesos). Los backends
LLM (Fase 2) se cargan perezosamente y caen al determinista si su
dependencia no está disponible.
"""
from __future__ import annotations

from typing import Protocol

from src.language.schema import Instruction


class InstructionParser(Protocol):
    """Contrato: texto crudo -> Instruction estructurada (sin estado de escena)."""

    def parse(self, text: str) -> Instruction: ...


def make_parser(backend: str = "deterministic", **kwargs) -> InstructionParser:
    """Crea un parser por nombre de backend.

    Args:
        backend: "deterministic" (default) | "llm_local" | "llm_api".
        **kwargs: opciones específicas del backend (p. ej. model=...).

    Raises:
        ValueError: si el backend no existe.
    """
    if backend == "deterministic":
        from src.language.backends.deterministic import DeterministicParser
        return DeterministicParser()
    if backend == "llm_local":
        from src.language.backends.llm_local import LLMLocalParser
        return LLMLocalParser(**kwargs)
    if backend == "llm_api":
        from src.language.backends.llm_api import LLMApiParser
        return LLMApiParser(**kwargs)
    raise ValueError(f"Backend de parser desconocido: {backend!r}")
```

```python
# src/language/backends/__init__.py
"""Backends del InstructionParser."""
```

```python
# src/language/backends/deterministic.py
"""Parser determinista basado en léxico controlado.

Reproducible, sin dependencias externas. Cubre el vocabulario controlado
(colores, formas, tamaños, relaciones espaciales, secuencias de 2 pasos).
"""
from __future__ import annotations

import re

from src.language import vocab
from src.language.schema import Instruction, SpatialRelation, TargetSpec

# separadores de pasos secuenciales (exp23)
_SEQ_SPLIT = re.compile(r"\b(?:and then|then|y luego|y despu[eé]s|luego)\b", re.I)


def _parse_target(text: str) -> TargetSpec:
    noun = next((n for n in vocab.NOUNS if n in text.lower()), None)
    return TargetSpec(
        color=vocab.normalize_color(text),
        shape=vocab.normalize_shape(text),
        size=vocab.normalize_size(text),
        raw_noun=noun,
    )


def _parse_single(text: str) -> Instruction:
    target = _parse_target(text)
    rel = vocab.normalize_relation(text)
    spatial = SpatialRelation(relation=rel) if rel else None
    # confianza baja si no se extrajo nada discriminativo
    conf = 1.0 if (not target.is_empty() or spatial) else 0.3
    return Instruction(
        raw_text=text,
        target=target,
        intent="pick",
        spatial=spatial,
        confidence=conf,
        backend="deterministic",
    )


class DeterministicParser:
    """Implementa InstructionParser con reglas/léxico."""

    def parse(self, text: str) -> Instruction:
        parts = [p.strip() for p in _SEQ_SPLIT.split(text) if p.strip()]
        if len(parts) > 1:
            steps = [_parse_single(p) for p in parts]
            return Instruction(
                raw_text=text,
                target=steps[0].target,
                intent="sequence",
                steps=steps,
                confidence=min(s.confidence for s in steps),
                backend="deterministic",
            )
        return _parse_single(text)
```

Añadir export en `src/language/__init__.py`:

```python
# src/language/__init__.py  (añadir tras los imports de schema)
from src.language.parser import InstructionParser, make_parser  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_language_parser.py -q`
Expected: PASS (todos los tests de parser/vocab).

- [ ] **Step 5: Commit**

```bash
git add src/language/parser.py src/language/backends/ src/language/__init__.py tests/test_language_parser.py
git commit -m "feat(language): parser determinista + factory make_parser"
```

---

### Task 4: Grounder determinista + CLIP opcional (`src/language/grounding.py`)

**Files:**
- Create: `src/language/grounding.py`
- Test: `tests/test_language_grounding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_language_grounding.py
"""Tests del Grounder (selección de target). Base 100% determinista."""
from src.language import make_parser
from src.language.grounding import Grounder
from src.language.schema import ObjectView


def _escena():
    # 3 objetos con atributos conocidos (como vendrían de metadatos de sim o CLIP)
    return [
        ObjectView(0, centroid=(-0.20, 0.0, 0.5), attributes={"color": "red", "shape": "cube", "size": "large"}),
        ObjectView(1, centroid=( 0.00, 0.0, 0.5), attributes={"color": "blue", "shape": "cube", "size": "small"}),
        ObjectView(2, centroid=( 0.20, 0.0, 0.5), attributes={"color": "red", "shape": "sphere", "size": "small"}),
    ]


def test_selecciona_por_color_y_forma():
    g = Grounder(method="attribute")
    instr = make_parser().parse("pick the red cube")
    res = g.ground(instr, _escena())
    assert res.target_obj_id == 0
    assert not res.ambiguous


def test_relacion_espacial_desempata():
    # dos rojos; "el rojo de la izquierda" -> obj_id 0 (x más negativo)
    g = Grounder(method="attribute")
    instr = make_parser().parse("dame el rojo de la izquierda")
    res = g.ground(instr, _escena())
    assert res.target_obj_id == 0


def test_nearest_por_profundidad():
    g = Grounder(method="attribute")
    objs = [
        ObjectView(0, centroid=(0.0, 0.0, 0.8), attributes={"color": "red"}),
        ObjectView(1, centroid=(0.1, 0.0, 0.4), attributes={"color": "red"}),
    ]
    instr = make_parser().parse("the nearest red object")
    res = g.ground(instr, objs)
    assert res.target_obj_id == 1   # menor z = más cercano


def test_ambiguedad_detectada():
    g = Grounder(method="attribute")
    objs = [
        ObjectView(0, centroid=(-0.1, 0, 0.5), attributes={"color": "red", "shape": "cube"}),
        ObjectView(1, centroid=( 0.1, 0, 0.5), attributes={"color": "red", "shape": "cube"}),
    ]
    instr = make_parser().parse("pick the red cube")
    res = g.ground(instr, objs)
    assert res.ambiguous          # dos candidatos empatados, sin relación espacial


def test_sin_match_devuelve_none():
    g = Grounder(method="attribute")
    instr = make_parser().parse("pick the green cylinder")
    res = g.ground(instr, _escena())
    assert res.target_obj_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_language_grounding.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.language.grounding'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/language/grounding.py
"""Grounder: asocia una Instruction a los objetos detectados de la escena.

Método "attribute" (default, determinista): puntúa por coincidencia de
atributos (color/forma/tamaño) y aplica la relación espacial sobre los
centroides. Método "clip_image" (opcional, lazy): rellena atributos con
CLIP cuando no vienen dados. La lógica de selección es compartida.
"""
from __future__ import annotations

from src.language.schema import GroundingResult, Instruction, ObjectView, TargetSpec

# umbral para considerar dos scores "empatados" (ambigüedad)
_TIE_EPS = 1e-6


def _attr_score(spec: TargetSpec, attrs: dict) -> float:
    """Fracción de atributos especificados que coinciden (0..1)."""
    checks = []
    for key in ("color", "shape", "size"):
        want = getattr(spec, key)
        if want is not None:
            checks.append(1.0 if attrs.get(key) == want else 0.0)
    if not checks:
        return 0.0
    return sum(checks) / len(checks)


def _apply_spatial(relation: str, candidates: list[ObjectView]) -> ObjectView | None:
    """Elige un candidato según la relación espacial (centroide = (x,y,z))."""
    if not candidates:
        return None
    if relation == "left_of":
        return min(candidates, key=lambda o: o.centroid[0])
    if relation == "right_of":
        return max(candidates, key=lambda o: o.centroid[0])
    if relation == "nearest":
        return min(candidates, key=lambda o: o.centroid[2])
    if relation == "farthest":
        return max(candidates, key=lambda o: o.centroid[2])
    if relation == "on_top":
        return max(candidates, key=lambda o: o.centroid[1])
    return None


class Grounder:
    """Selecciona el objeto target descrito por una Instruction."""

    def __init__(self, method: str = "attribute", clip_model: str = "openai/clip-vit-base-patch32"):
        self.method = method
        self.clip_model = clip_model
        self._clip = None  # lazy

    def _ensure_attributes(self, objects: list[ObjectView], rgb=None) -> list[ObjectView]:
        """Si method=clip_image y faltan atributos, los rellena con CLIP.

        En method=attribute se asume que los ObjectView ya traen atributos
        (de metadatos de simulación o de una pasada previa).
        """
        if self.method != "clip_image" or rgb is None:
            return objects
        from src.language._clip import clip_attributes  # lazy, opcional
        if self._clip is None:
            self._clip = clip_attributes  # función cacheada en el módulo
        for o in objects:
            if not o.attributes and o.bbox is not None:
                o.attributes = self._clip(rgb, o.bbox, self.clip_model)
        return objects

    def ground(self, instruction: Instruction, objects: list[ObjectView], rgb=None, K=None) -> GroundingResult:
        objects = self._ensure_attributes(objects, rgb)
        spec = instruction.target

        scores = {o.obj_id: _attr_score(spec, o.attributes) for o in objects}
        max_score = max(scores.values(), default=0.0)

        if max_score <= 0.0:
            return GroundingResult(
                target_obj_id=None, scores=scores, method=self.method,
                rejected=[o.obj_id for o in objects], ambiguous=False,
            )

        winners = [o for o in objects if abs(scores[o.obj_id] - max_score) <= _TIE_EPS]
        rejected = [o.obj_id for o in objects if o not in winners]

        if len(winners) == 1:
            return GroundingResult(
                target_obj_id=winners[0].obj_id, scores=scores,
                method=self.method, rejected=rejected, ambiguous=False,
            )

        # desempate por relación espacial
        if instruction.spatial is not None:
            chosen = _apply_spatial(instruction.spatial.relation, winners)
            if chosen is not None:
                return GroundingResult(
                    target_obj_id=chosen.obj_id, scores=scores,
                    method="spatial",
                    rejected=[o.obj_id for o in objects if o.obj_id != chosen.obj_id],
                    ambiguous=False,
                )

        # empate irresoluble -> ambiguo (se devuelve el primero como sugerencia)
        return GroundingResult(
            target_obj_id=winners[0].obj_id, scores=scores, method=self.method,
            rejected=rejected, ambiguous=True,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_language_grounding.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/language/grounding.py tests/test_language_grounding.py
git commit -m "feat(language): Grounder determinista con desempate espacial"
```

---

### Task 5: Helper CLIP opcional (`src/language/_clip.py`)

**Files:**
- Create: `src/language/_clip.py`
- Test: `tests/test_language_grounding.py` (un test marcado `slow`)

- [ ] **Step 1: Añadir marcador `slow` en `pyproject.toml`**

En `[tool.pytest.ini_options]`, sección `markers`, añadir tras la línea `integration`:

```toml
    "slow: requiere pesos/red (CLIP o LLM); se omite en la suite base",
```

- [ ] **Step 2: Write the failing test (añadir a test_language_grounding.py)**

```python
# tests/test_language_grounding.py  (añadir)
import pytest


@pytest.mark.slow
def test_clip_attributes_clasifica_color():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    import numpy as np
    from src.language._clip import clip_attributes
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[..., 0] = 220   # parche rojo
    attrs = clip_attributes(rgb, bbox=(0, 0, 64, 64))
    assert attrs.get("color") == "red"
```

- [ ] **Step 3: Run test to verify it fails (o se omite)**

Run: `python -m pytest tests/test_language_grounding.py -q -m slow`
Expected: FAIL con `ModuleNotFoundError: src.language._clip` (si transformers está; si no, SKIP). En la suite base (`-m "not slow"`) este test se omite.

- [ ] **Step 4: Write minimal implementation**

```python
# src/language/_clip.py
"""Clasificación zero-shot de atributos con CLIP (opcional, lazy).

Recorta el bbox de un objeto y compara contra prompts de color/forma/tamaño
con CLIP image-text (exp24). Sólo se importa si method="clip_image".
"""
from __future__ import annotations

from functools import lru_cache

from src.language.vocab import COLORS, SHAPES, SIZES


def _crop(rgb, bbox):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    return rgb[y1:y2, x1:x2]


@lru_cache(maxsize=2)
def _load_clip(model_name: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    return model, proc, torch


def _best_label(crop, table, model_name):
    model, proc, torch = _load_clip(model_name)
    labels = list(table.keys())
    prompts = [f"a photo of a {l} object" for l in labels]
    inputs = proc(text=prompts, images=crop, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image.softmax(dim=1)[0]
    return labels[int(logits.argmax())]


def clip_attributes(rgb, bbox, model_name: str = "openai/clip-vit-base-patch32") -> dict:
    """Devuelve {"color","shape","size"} estimados por CLIP para un objeto."""
    crop = _crop(rgb, bbox)
    return {
        "color": _best_label(crop, COLORS, model_name),
        "shape": _best_label(crop, SHAPES, model_name),
        "size": _best_label(crop, SIZES, model_name),
    }
```

- [ ] **Step 5: Run base suite (sin slow) y commit**

Run: `python -m pytest tests/test_language_grounding.py -q -m "not slow"`
Expected: PASS (los 5 deterministas; el `slow` omitido).

```bash
git add src/language/_clip.py tests/test_language_grounding.py pyproject.toml
git commit -m "feat(language): clasificación de atributos con CLIP (opcional, lazy)"
```

---

### Task 6: Integración en el pipeline (`src/pipeline.py`)

**Files:**
- Modify: `src/pipeline.py` (PipelineConfig, PipelineResult, BinPickingPipeline)
- Test: `tests/test_pipeline_language.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_language.py
"""Integración del lenguaje en BinPickingPipeline (con poses mock)."""
import numpy as np
from src.pipeline import BinPickingPipeline, PipelineConfig, PoseResult


def _pose(obj_id, x, color, shape):
    T = np.eye(4); T[0, 3] = x; T[2, 3] = 0.5
    p = PoseResult(obj_id=obj_id, R=np.eye(3), t=np.array([x, 0, 0.5]),
                   score=0.9, T=T)
    p.attributes = {"color": color, "shape": shape}   # metadatos de escena
    return p


def test_objectviews_desde_poses():
    cfg = PipelineConfig(language_enabled=True)
    pipe = BinPickingPipeline(cfg)
    poses = [_pose(0, -0.2, "red", "cube"), _pose(1, 0.2, "blue", "cube")]
    views = pipe._poses_to_views(poses)
    assert len(views) == 2
    assert views[0].centroid[0] == -0.2
    assert views[0].attributes["color"] == "red"


def test_select_target_por_instruccion():
    cfg = PipelineConfig(language_enabled=True)
    pipe = BinPickingPipeline(cfg)
    poses = [_pose(0, -0.2, "red", "cube"), _pose(1, 0.2, "blue", "cube")]
    selected, grounding, instruction = pipe.select_target(poses, "pick the red cube")
    assert [p.obj_id for p in selected] == [0]
    assert grounding.target_obj_id == 0
    assert instruction.target.color == "red"


def test_sin_instruccion_no_filtra():
    cfg = PipelineConfig(language_enabled=True)
    pipe = BinPickingPipeline(cfg)
    poses = [_pose(0, -0.2, "red", "cube"), _pose(1, 0.2, "blue", "cube")]
    selected, grounding, instruction = pipe.select_target(poses, None)
    assert len(selected) == 2 and grounding is None and instruction is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_language.py -q`
Expected: FAIL con `AttributeError: 'PipelineConfig' has no attribute 'language_enabled'`.

- [ ] **Step 3: Write minimal implementation**

En `PipelineConfig` (tras `device`), añadir:

```python
    # Lenguaje natural (PLN)
    language_enabled: bool = False
    parser_backend: str = "deterministic"   # deterministic | llm_local | llm_api
    grounder_method: str = "attribute"      # attribute | clip_image
    ambiguity_tolerant: bool = True         # si ambiguo, conserva candidatos ordenados
```

En `PipelineResult`, añadir campos opcionales (tras `timing`):

```python
    instruction: Optional["object"] = None   # src.language.schema.Instruction
    grounding: Optional["object"] = None     # src.language.schema.GroundingResult
```

En `BinPickingPipeline`, añadir métodos:

```python
    def _poses_to_views(self, poses):
        """Convierte PoseResult -> ObjectView para el Grounder."""
        from src.language.schema import ObjectView
        views = []
        for p in poses:
            attrs = getattr(p, "attributes", {}) or {}
            bbox = tuple(p.bbox) if getattr(p, "bbox", None) is not None else None
            views.append(ObjectView(
                obj_id=p.obj_id,
                centroid=(float(p.t[0]), float(p.t[1]), float(p.t[2])),
                attributes=attrs, bbox=bbox,
            ))
        return views

    def select_target(self, poses, instruction):
        """Aplica lenguaje natural para seleccionar el/los target(s).

        Returns:
            (poses_seleccionadas, GroundingResult|None, Instruction|None)
        """
        if not instruction:
            return poses, None, None
        from src.language import make_parser
        from src.language.grounding import Grounder
        parser = make_parser(self.config.parser_backend)
        grounder = Grounder(method=self.config.grounder_method)
        instr = parser.parse(instruction)
        views = self._poses_to_views(poses)
        result = grounder.ground(instr, views)
        by_id = {p.obj_id: p for p in poses}
        if result.target_obj_id is None:
            return [], result, instr
        if result.ambiguous and self.config.ambiguity_tolerant:
            ordered = sorted(poses, key=lambda p: result.scores.get(p.obj_id, 0.0),
                             reverse=True)
            return ordered, result, instr
        return [by_id[result.target_obj_id]], result, instr
```

Modificar `run(...)` para aceptar `instruction`:

```python
    def run(self, rgb, depth, K, masks=None, instruction=None):
        if not self._initialized:
            raise RuntimeError("Call initialize() first")
        timing = {}
        t0 = time.time()
        poses = self.estimate_poses(rgb, depth, K, masks)
        timing["pose_estimation"] = time.time() - t0

        instr_obj = grounding = None
        if instruction and self.config.language_enabled:
            t0 = time.time()
            poses, grounding, instr_obj = self.select_target(poses, instruction)
            timing["language_grounding"] = time.time() - t0

        t0 = time.time()
        grasps = self.plan_grasps(poses)
        timing["grasp_planning"] = time.time() - t0
        best = grasps[0] if grasps else None
        timing["total"] = sum(timing.values())
        return PipelineResult(poses=poses, grasps=grasps, best_grasp=best,
                              timing=timing, instruction=instr_obj, grounding=grounding)
```

Nota: `PoseResult` no declara `attributes`; el test lo asigna dinámicamente y
`_poses_to_views` usa `getattr(..., {})`. Para hacerlo explícito, añadir a
`PoseResult` el campo opcional:

```python
    attributes: dict = field(default_factory=dict)  # color/shape/size si se conocen
```

- [ ] **Step 4: Run tests (nuevo + no-regresión)**

Run: `python -m pytest tests/test_pipeline_language.py tests/test_pipeline.py -q`
Expected: PASS (todos; el comportamiento previo de `test_pipeline.py` intacto).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline_language.py
git commit -m "feat(pipeline): run(instruction=...) selecciona target por lenguaje natural"
```

---

### Task 7: CLI de demo (`experiments/run_pick_language.py`)

**Files:**
- Create: `experiments/run_pick_language.py`
- Test: `tests/test_pipeline_language.py` (test de import/CLI args, sin CoppeliaSim)

- [ ] **Step 1: Write the failing test (añadir)**

```python
# tests/test_pipeline_language.py  (añadir)
import importlib, sys
from pathlib import Path


def test_cli_language_parsea_args(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    mod = importlib.import_module("experiments.run_pick_language")
    args = mod.build_parser().parse_args(
        ["--instruction", "pick the red cube", "--scene", "multi", "--dry-run"]
    )
    assert args.instruction == "pick the red cube"
    assert args.scene == "multi"
    assert args.dry_run is True


def test_cli_dry_run_ejecuta_grounding(capsys):
    from experiments.run_pick_language import run_dry
    code = run_dry("dame el cubo rojo de la izquierda")
    out = capsys.readouterr().out
    assert code == 0
    assert "target" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_language.py -q -k cli`
Expected: FAIL con `ModuleNotFoundError: experiments.run_pick_language`.

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Demo de bin picking guiado por lenguaje natural.

Ejecuta el pipeline end-to-end seleccionando el objeto descrito por una
instrucción en lenguaje natural. Con --dry-run no requiere CoppeliaSim:
muestra el parsing + grounding sobre una escena sintética fija.

Ejemplos:
    python experiments/run_pick_language.py --instruction "dame el cubo rojo" --dry-run
    python experiments/run_pick_language.py --instruction "pick the blue sphere" --scene clutter --render
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.language import make_parser            # noqa: E402
from src.language.grounding import Grounder     # noqa: E402
from src.language.schema import ObjectView      # noqa: E402

RESULTS = REPO / "experiments/results/language_pick"


def _escena_demo() -> list[ObjectView]:
    """Escena sintética fija de 3 objetos (coherente con exp16/24)."""
    return [
        ObjectView(0, (-0.20, 0.0, 0.5), {"color": "red", "shape": "cube", "size": "large"}),
        ObjectView(1, (0.00, 0.0, 0.5), {"color": "blue", "shape": "cube", "size": "small"}),
        ObjectView(2, (0.20, 0.0, 0.5), {"color": "red", "shape": "sphere", "size": "small"}),
    ]


def run_dry(instruction: str, backend: str = "deterministic") -> int:
    """Parsing + grounding sobre la escena demo; imprime y guarda JSON."""
    parser = make_parser(backend)
    grounder = Grounder(method="attribute")
    instr = parser.parse(instruction)
    objs = _escena_demo()
    res = grounder.ground(instr, objs)
    payload = {
        "instruction": instruction,
        "parsed": {
            "color": instr.target.color, "shape": instr.target.shape,
            "size": instr.target.size, "intent": instr.intent,
            "spatial": instr.spatial.relation if instr.spatial else None,
            "backend": instr.backend,
        },
        "grounding": {
            "target_obj_id": res.target_obj_id, "method": res.method,
            "ambiguous": res.ambiguous, "scores": res.scores,
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "last_dry_run.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bin picking guiado por lenguaje natural")
    p.add_argument("--instruction", required=True, help='p.ej. "dame el cubo rojo"')
    p.add_argument("--parser-backend", default="deterministic",
                   choices=["deterministic", "llm_local", "llm_api"])
    p.add_argument("--scene", default="multi", choices=["multi", "clutter"])
    p.add_argument("--render", action="store_true", help="render del grounding (requiere sim)")
    p.add_argument("--dry-run", action="store_true",
                   help="solo parsing + grounding, sin CoppeliaSim")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        return run_dry(args.instruction, args.parser_backend)
    # Ruta E2E con CoppeliaSim: reusa la escena multi-objeto y el pipeline.
    # Se delega a la batería de pick existente; aquí solo el grounding decide
    # el target. Requiere CoppeliaSim en localhost:23000.
    from experiments.run_pick_battery import run_language_pick  # implementado en sim
    return run_language_pick(instruction=args.instruction, scene=args.scene,
                             parser_backend=args.parser_backend, render=args.render)


if __name__ == "__main__":
    raise SystemExit(main())
```

Nota: `run_language_pick` en `run_pick_battery.py` es el enganche con la sim. Si
no existe aún, créalo como wrapper fino sobre la batería de pick existente que
acepte el `instruction`/`scene` y use `BinPickingPipeline.run(instruction=...)`.
Para este plan, el camino testeado y exigido es `--dry-run`; la ruta sim queda
cubierta por el marcador `integration` y se valida manualmente con CoppeliaSim.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pipeline_language.py -q`
Expected: PASS (incluye los tests CLI con `--dry-run`).

- [ ] **Step 5: Commit**

```bash
git add experiments/run_pick_language.py tests/test_pipeline_language.py
git commit -m "feat(cli): demo run_pick_language con grounding (--dry-run sin sim)"
```

---

### Task 8: Tab de dashboard "🗣️ Lenguaje natural"

**Files:**
- Modify: `dashboard.py`
- Test: `tests/test_dashboard_language.py` (con `streamlit.testing.AppTest`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard_language.py
"""Smoke test del tab de PLN del dashboard con AppTest."""
import pytest

pytest.importorskip("streamlit")


def test_render_language_tab_no_revienta():
    from dashboard import render_language_tab   # función pura, sin st.run completo
    payload = render_language_tab("dame el cubo rojo de la izquierda", run=True)
    assert payload["grounding"]["target_obj_id"] is not None
    assert payload["parsed"]["color"] == "red"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dashboard_language.py -q`
Expected: FAIL con `ImportError: cannot import name 'render_language_tab'`.

- [ ] **Step 3: Write minimal implementation**

En `dashboard.py`, añadir una función reutilizable (lógica separada del render
Streamlit para poder testearla) y su sección:

```python
def render_language_tab(instruction: str, run: bool = False) -> dict:
    """Lógica del tab de lenguaje natural. Devuelve el payload de grounding.

    Separada de los widgets para ser testeable sin servidor Streamlit.
    """
    from src.language import make_parser
    from src.language.grounding import Grounder
    from src.language.schema import ObjectView
    objs = [
        ObjectView(0, (-0.20, 0.0, 0.5), {"color": "red", "shape": "cube", "size": "large"}),
        ObjectView(1, (0.00, 0.0, 0.5), {"color": "blue", "shape": "cube", "size": "small"}),
        ObjectView(2, (0.20, 0.0, 0.5), {"color": "red", "shape": "sphere", "size": "small"}),
    ]
    instr = make_parser("deterministic").parse(instruction)
    res = Grounder(method="attribute").ground(instr, objs)
    return {
        "parsed": {"color": instr.target.color, "shape": instr.target.shape,
                   "size": instr.target.size,
                   "spatial": instr.spatial.relation if instr.spatial else None},
        "grounding": {"target_obj_id": res.target_obj_id, "scores": res.scores,
                      "ambiguous": res.ambiguous, "method": res.method},
    }
```

Y la sección Streamlit (donde se registran las pestañas/secciones del sidebar),
siguiendo el patrón de las secciones existentes:

```python
    # --- Sección: Lenguaje natural (PLN) ---
    st.header("🗣️ Bin picking guiado por lenguaje natural")
    st.markdown(
        "Escribe una instrucción y observa el *parsing* y la selección de objetivo "
        "(grounding). Consolida las exploraciones VLA/CLIP (exp16–26)."
    )
    instr_text = st.text_input("Instrucción", value="dame el cubo rojo de la izquierda")
    if st.button("Interpretar y seleccionar"):
        payload = render_language_tab(instr_text, run=True)
        st.json(payload["parsed"])
        tgt = payload["grounding"]["target_obj_id"]
        if tgt is None:
            st.warning("Ningún objeto coincide con la instrucción.")
        elif payload["grounding"]["ambiguous"]:
            st.info(f"Ambiguo — sugerido objeto #{tgt}. Añade una relación espacial.")
        else:
            st.success(f"Objeto seleccionado: #{tgt}  (método: {payload['grounding']['method']})")
        st.bar_chart(payload["grounding"]["scores"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dashboard_language.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard.py tests/test_dashboard_language.py
git commit -m "feat(dashboard): tab interactivo de lenguaje natural (PLN)"
```

---

## FASE 2 — Lenguaje libre con LLM local

### Task 9: Backend LLM local con fallback (`backends/llm_local.py`)

**Files:**
- Create: `src/language/backends/llm_local.py`
- Test: `tests/test_language_llm_fallback.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_language_llm_fallback.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_language_llm_fallback.py -q`
Expected: FAIL con `ModuleNotFoundError: src.language.backends.llm_local`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/language/backends/llm_local.py
"""Parser con LLM open-source local (Ollama/MLX), con fallback determinista.

Estrategia: prompt few-shot que pide un JSON con el esquema de Instruction;
se valida y se mapea a dataclasses. Si Ollama no está disponible o el JSON no
valida -> se delega en DeterministicParser (degradación elegante).
"""
from __future__ import annotations

import json
from typing import Optional

from src.language import vocab
from src.language.backends.deterministic import DeterministicParser
from src.language.schema import Instruction, SpatialRelation, TargetSpec

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5:3b-instruct"

_SYSTEM = (
    "Eres un parser de instrucciones de robótica. Devuelve SOLO un JSON con las "
    "claves: intent (pick|pick_then_place|sequence), color, shape, size, relation. "
    "Usa valores en inglés o null. No añadas texto fuera del JSON."
)
_FEWSHOT = '{"intent": "pick", "color": "red", "shape": "cube", "size": null, "relation": "left_of"}'


def _ollama_available(host: str) -> bool:
    """True si hay un servidor Ollama respondiendo en host."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{host}/api/tags", timeout=0.5) as r:
            return r.status == 200
    except Exception:
        return False


def _ollama_generate(host: str, model: str, prompt: str) -> str:
    """Llama a /api/generate y devuelve la respuesta cruda."""
    import urllib.request
    body = json.dumps({"model": model, "prompt": prompt, "system": _SYSTEM,
                       "stream": False, "format": "json"}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode()).get("response", "")


def _json_to_instruction(raw_text: str, data: dict, model: str) -> Optional[Instruction]:
    """Mapea el JSON del LLM a Instruction (normalizando con el léxico)."""
    try:
        target = TargetSpec(
            color=vocab.normalize_color(str(data.get("color") or "")),
            shape=vocab.normalize_shape(str(data.get("shape") or "")),
            size=vocab.normalize_size(str(data.get("size") or "")),
        )
        rel = data.get("relation")
        spatial = SpatialRelation(relation=vocab.normalize_relation(str(rel))) \
            if rel and vocab.normalize_relation(str(rel)) else None
        intent = data.get("intent") or "pick"
        if target.is_empty() and not spatial:
            return None  # nada útil -> fallback
        return Instruction(raw_text=raw_text, target=target, intent=intent,
                           spatial=spatial, confidence=0.9,
                           backend=f"llm_local:{model}")
    except Exception:
        return None


class LLMLocalParser:
    """InstructionParser con LLM local y fallback determinista."""

    def __init__(self, model: str = _DEFAULT_MODEL, host: str = _DEFAULT_HOST):
        self.model = model
        self.host = host
        self._fallback = DeterministicParser()

    def parse(self, text: str) -> Instruction:
        if not _ollama_available(self.host):
            return self._fallback.parse(text)
        prompt = f"Instrucción: {text}\nEjemplo de salida: {_FEWSHOT}\nJSON:"
        try:
            raw = _ollama_generate(self.host, self.model, prompt)
            data = json.loads(raw)
        except Exception:
            return self._fallback.parse(text)
        instr = _json_to_instruction(text, data, self.model)
        return instr if instr is not None else self._fallback.parse(text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_language_llm_fallback.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/language/backends/llm_local.py tests/test_language_llm_fallback.py
git commit -m "feat(language): backend LLM local (Ollama) con fallback determinista"
```

---

### Task 10: Backend API stub + extras opcionales (`backends/llm_api.py`, `pyproject.toml`)

**Files:**
- Create: `src/language/backends/llm_api.py`
- Modify: `pyproject.toml` (extra `language-llm`)
- Test: `tests/test_language_llm_fallback.py` (añadir)

- [ ] **Step 1: Write the failing test (añadir)**

```python
# tests/test_language_llm_fallback.py  (añadir)
def test_llm_api_sin_clave_cae_a_determinista(monkeypatch):
    monkeypatch.delenv("LANGUAGE_API_KEY", raising=False)
    parser = make_parser("llm_api")
    instr = parser.parse("pick the yellow box")
    assert instr.target.color == "yellow"
    assert "deterministic" in instr.backend


def test_llm_api_no_disponible_flag(monkeypatch):
    import src.language.backends.llm_api as m
    monkeypatch.delenv("LANGUAGE_API_KEY", raising=False)
    assert m.LLMApiParser().available() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_language_llm_fallback.py -q -k api`
Expected: FAIL con `ModuleNotFoundError: src.language.backends.llm_api`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/language/backends/llm_api.py
"""Parser con API de LLM (punto de extensión opcional).

NO se usa por defecto y NO contiene claves. Lee la configuración de entorno:
    LANGUAGE_API_PROVIDER  (informativo)
    LANGUAGE_API_KEY       (si falta -> backend no disponible -> fallback determinista)
Mantiene la misma interfaz que los demás parsers.
"""
from __future__ import annotations

import os
from typing import Optional

from src.language.backends.deterministic import DeterministicParser
from src.language.schema import Instruction


class LLMApiParser:
    """InstructionParser vía API remota; fallback determinista si no hay clave."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or os.environ.get("LANGUAGE_API_PROVIDER", "")
        self.api_key = os.environ.get("LANGUAGE_API_KEY", "")
        self._fallback = DeterministicParser()

    def available(self) -> bool:
        return bool(self.api_key)

    def parse(self, text: str) -> Instruction:
        if not self.available():
            return self._fallback.parse(text)
        # Punto de extensión: implementar la llamada al proveedor elegido y
        # mapear su salida JSON a Instruction (mismo esquema que llm_local).
        # Por defecto, hasta implementar un proveedor concreto, se delega.
        return self._fallback.parse(text)
```

- [ ] **Step 4: Añadir extra opcional en `pyproject.toml`**

En `[project.optional-dependencies]`, añadir:

```toml
# Parser de lenguaje natural con LLM local/remoto (Fase 2, opcional).
language-llm = [
    "ollama>=0.3",
]
```

- [ ] **Step 5: Run tests y commit**

Run: `python -m pytest tests/test_language_llm_fallback.py -q`
Expected: PASS (5 passed en total).

```bash
git add src/language/backends/llm_api.py pyproject.toml tests/test_language_llm_fallback.py
git commit -m "feat(language): backend API enchufable (stub) + extra opcional language-llm"
```

---

## FASE 3 — Documentación y comparativa SOTA

### Task 11: Documento de feature + comparativa SOTA

**Files:**
- Create: `docs/LENGUAJE_NATURAL.md`
- Create: `docs/COMPARATIVA_SOTA_LENGUAJE.md`
- Modify: `README.md` (subsección)
- Modify: `docs/exploraciones/04_vla_lite_clip.md` (nota de consolidación)

- [ ] **Step 1: Escribir `docs/LENGUAJE_NATURAL.md`**

Contenido (secciones): Motivación; Arquitectura (diagrama Parser→Grounder→Pipeline);
API (`pipeline.run(instruction=...)`, `make_parser`, `Grounder.ground`); Backends
(determinista / llm_local / llm_api) y cómo instalar el extra
`pip install -e ".[language-llm]"` + `ollama pull qwen2.5:3b-instruct`; Ejemplos
ES/EN; Tabla consolidada de resultados exp16–26 (selection accuracy por atributo
y nº de objetos); Limitaciones; cómo correr la demo (`run_pick_language.py
--dry-run`) y el tab del dashboard.

- [ ] **Step 2: Escribir `docs/COMPARATIVA_SOTA_LENGUAJE.md`**

Tabla comparativa frente a CLIPort, VoxPoser, OWL-ViT/CLIP-Fields, SayCan en ejes:
licencia (open-source/Apache vs. restrictiva), hardware (M1 local vs. clúster/GPU),
re-entrenamiento por vocabulario (sí/no), interpretabilidad (caja blanca vs. negra),
accuracy de selección reportada. Texto que argumenta la posición competitiva:
**pipeline 100% open-license, corre en portátil, grounding determinista+CLIP a
98–100%, comprensión LLM enchufable**. Cada cifra de terceros con su cita de paper.

> ⚠️ No inventar cifras. Citar de los papers originales; si un dato no se conoce
> con certeza, marcar "n/d" con nota.

- [ ] **Step 3: Subsección en `README.md`**

Añadir "Natural-language bin picking" con quickstart de 3 líneas:

```python
from src.pipeline import BinPickingPipeline, PipelineConfig
pipe = BinPickingPipeline(PipelineConfig(language_enabled=True)); pipe.initialize()
result = pipe.run(rgb, depth, K, instruction="dame el cubo rojo de la izquierda")
```

y enlace a `docs/LENGUAJE_NATURAL.md`.

- [ ] **Step 4: Verificar que los docs no rompen nada (lint de enlaces básico)**

Run: `python -m pytest -q -m "not slow and not integration"`
Expected: PASS (toda la suite base verde — los docs no afectan tests).

- [ ] **Step 5: Commit**

```bash
git add docs/LENGUAJE_NATURAL.md docs/COMPARATIVA_SOTA_LENGUAJE.md README.md docs/exploraciones/04_vla_lite_clip.md
git commit -m "docs: feature de lenguaje natural + comparativa SOTA open-source"
```

---

### Task 12: Entrada de memoria del proyecto

**Files:**
- Create: `~/.claude/projects/-Users-giocrisraigodoy-Documents-MATLAB-TFM/memory/tfm-lenguaje-natural.md`
- Modify: `~/.claude/projects/.../memory/MEMORY.md` (índice)

- [ ] **Step 1: Crear la memoria**

```markdown
---
name: tfm-lenguaje-natural
description: feature PLN open-source — parser determinista + Grounder + LLM local; consolida exp16-26 en el pipeline
metadata:
  type: project
---

Feature de bin picking guiado por lenguaje natural añadida en rama
`feat/lenguaje-natural-pln`. Arquitectura: `src/language/` con Parser
(determinista default, LLM local Ollama en Fase 2 con fallback) + Grounder
(attribute determinista + CLIP opcional). Integrado en
`BinPickingPipeline.run(instruction=...)`, CLI `run_pick_language.py --dry-run`,
tab de dashboard, docs `LENGUAJE_NATURAL.md` + comparativa SOTA. Consolida las
exploraciones exp16-26. Ver [[tfm-docs-ubicacion]] y [[tfm-dashboard]].
```

- [ ] **Step 2: Añadir línea al índice `MEMORY.md`**

```
- [Lenguaje natural (PLN)](tfm-lenguaje-natural.md) — feature open-source: parser determinista + Grounder + LLM local; consolida exp16-26
```

- [ ] **Step 3: Commit (solo si el repo versiona la memoria; si no, omitir)**

La memoria vive fuera del repo (`~/.claude/...`), no se commitea. Solo crear los ficheros.

---

## Cierre

- [ ] Ejecutar la suite base completa: `python -m pytest -q -m "not slow and not integration"` → todo verde.
- [ ] (Opcional, con red/pesos) `python -m pytest -q -m slow` para validar CLIP/LLM.
- [ ] (Opcional, con CoppeliaSim) validar la ruta E2E real de `run_pick_language.py`.
- [ ] Invocar `superpowers:finishing-a-development-branch` para decidir merge/PR.
```
