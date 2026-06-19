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
