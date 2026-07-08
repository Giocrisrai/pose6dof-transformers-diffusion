"""Parser determinista basado en léxico controlado.

Reproducible, sin dependencias externas. Cubre el vocabulario controlado
(colores, formas, tamaños, relaciones espaciales, secuencias de 2 pasos).
"""
from __future__ import annotations

import re

from src.language import vocab
from src.language.schema import Instruction, SpatialRelation, TargetSpec

# separadores de pasos secuenciales (exp23)
# Nota: "then"/"luego" solos pueden sobre-dividir, pero es aceptable para el vocabulario controlado.
_SEQ_SPLIT = re.compile(r"\b(?:and then|then|y luego|y despu[eé]s|luego)\b", re.I)


def _parse_target(text: str) -> TargetSpec:
    noun = next((n for n in vocab.NOUNS if re.search(rf"\b{re.escape(n)}\b", text.lower())), None)
    return TargetSpec(
        color=vocab.normalize_color(text),  # type: ignore[arg-type]
        shape=vocab.normalize_shape(text),  # type: ignore[arg-type]
        size=vocab.normalize_size(text),  # type: ignore[arg-type]
        raw_noun=noun,
    )


def _parse_single(text: str) -> Instruction:
    target = _parse_target(text)
    rel = vocab.normalize_relation(text)
    spatial = SpatialRelation(relation=rel) if rel else None  # type: ignore[arg-type]
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
        # Entrada vacía o de una sola parte: ambas caen directamente a _parse_single.
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
