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


def _ollama_generate(host: str, model: str, prompt: str, timeout: float = 15.0) -> str:
    """Llama a /api/generate y devuelve la respuesta cruda."""
    import urllib.request
    body = json.dumps({"model": model, "prompt": prompt, "system": _SYSTEM,
                       "stream": False, "format": "json"}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()).get("response", "")


def _json_to_instruction(raw_text: str, data: dict, model: str) -> Optional[Instruction]:
    """Mapea el JSON del LLM a Instruction (normalizando con el léxico)."""
    try:
        target = TargetSpec(
            color=vocab.normalize_color(str(data.get("color") or "")),  # type: ignore[arg-type]
            shape=vocab.normalize_shape(str(data.get("shape") or "")),  # type: ignore[arg-type]
            size=vocab.normalize_size(str(data.get("size") or "")),  # type: ignore[arg-type]
        )
        rel_raw = data.get("relation")
        rel = vocab.normalize_relation(str(rel_raw)) if rel_raw else None
        spatial = SpatialRelation(relation=rel) if rel else None  # type: ignore[arg-type]
        intent = data.get("intent") or "pick"
        if intent not in ("pick", "pick_then_place", "sequence"):
            intent = "pick"
        if target.is_empty() and not spatial:
            return None  # nada útil -> fallback
        return Instruction(raw_text=raw_text, target=target, intent=intent,  # type: ignore[arg-type]
                           spatial=spatial, confidence=0.9,
                           backend=f"llm_local:{model}")
    except Exception:
        return None


class LLMLocalParser:
    """InstructionParser con LLM local y fallback determinista."""

    def __init__(self, model: str = _DEFAULT_MODEL, host: str = _DEFAULT_HOST,
                 gen_timeout: float = 15.0):
        self.model = model
        self.host = host
        self.gen_timeout = gen_timeout
        self._fallback = DeterministicParser()

    def parse(self, text: str) -> Instruction:
        if not _ollama_available(self.host):
            return self._fallback.parse(text)
        prompt = f"Instrucción: {text}\nEjemplo de salida: {_FEWSHOT}\nJSON:"
        try:
            raw = _ollama_generate(self.host, self.model, prompt, self.gen_timeout)
            data = json.loads(raw)
        except Exception:
            return self._fallback.parse(text)
        instr = _json_to_instruction(text, data, self.model)
        return instr if instr is not None else self._fallback.parse(text)
