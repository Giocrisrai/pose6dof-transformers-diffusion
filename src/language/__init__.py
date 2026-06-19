"""Capacidad de bin picking guiado por lenguaje natural (PLN).

API pública estable:
    from src.language import make_parser, Grounder, Instruction
"""
from src.language.schema import (  # noqa: F401
    TargetSpec, SpatialRelation, Instruction, GroundingResult, ObjectView,
)
from src.language.parser import InstructionParser, make_parser  # noqa: F401
from src.language.grounding import Grounder  # noqa: F401
