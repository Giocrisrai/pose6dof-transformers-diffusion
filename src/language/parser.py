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
