"""Logging estructurado para el pipeline TFM.

Sistema unificado de logging con:
- Formateo consistente con timestamp + nivel + módulo
- Output a consola y/o archivo
- Niveles configurables por env var TFM_LOG_LEVEL
- Soporte para formato JSON estructurado (para parseo automatizado)

Uso:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting evaluation")
    logger.warning("...")
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


COLORS = {
    'DEBUG': '\033[36m',      # cyan
    'INFO': '\033[32m',       # green
    'WARNING': '\033[33m',    # yellow
    'ERROR': '\033[31m',      # red
    'CRITICAL': '\033[35m',   # magenta
    'RESET': '\033[0m',
}


class ColoredFormatter(logging.Formatter):
    """Formateador con colores ANSI por nivel."""

    def format(self, record):
        color = COLORS.get(record.levelname, '')
        reset = COLORS['RESET']
        record.levelname_colored = f"{color}{record.levelname:8}{reset}"
        return super().format(record)


_loggers_configured = set()


def get_logger(name: str = "tfm", level: Optional[str] = None,
               log_file: Optional[Path] = None,
               format_json: bool = False) -> logging.Logger:
    """Devuelve un logger configurado con formato unificado.

    Args:
        name: nombre del logger (típicamente __name__).
        level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'. Si None, lee TFM_LOG_LEVEL
               (default INFO).
        log_file: ruta opcional a fichero de log.
        format_json: si True, formato JSON estructurado para parseo automático.

    Returns:
        Logger configurado.
    """
    if level is None:
        level = os.environ.get("TFM_LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger(name)

    # Evitar configurar el mismo logger varias veces
    if name in _loggers_configured:
        return logger

    logger.setLevel(level)

    # Handler de consola
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)

    if format_json:
        fmt = '{"ts":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":"%(message)s"}'
    elif sys.stderr.isatty():
        fmt = '%(asctime)s | %(levelname_colored)s | %(name)s | %(message)s'
        console.setFormatter(ColoredFormatter(fmt, datefmt='%H:%M:%S'))
    else:
        fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'

    if not format_json and not sys.stderr.isatty():
        console.setFormatter(logging.Formatter(fmt, datefmt='%H:%M:%S'))
    elif format_json:
        console.setFormatter(logging.Formatter(fmt, datefmt='%Y-%m-%dT%H:%M:%S'))

    logger.addHandler(console)

    # Handler de archivo opcional
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        ))
        logger.addHandler(fh)

    logger.propagate = False
    _loggers_configured.add(name)
    return logger


def log_run_metadata(logger: logging.Logger, run_name: str, metadata: dict) -> None:
    """Loguea metadata estructurada de un experimento (datos, hiperparámetros, etc)."""
    logger.info("─" * 60)
    logger.info(f"Run: {run_name}")
    logger.info("─" * 60)
    for k, v in metadata.items():
        logger.info(f"  {k:25} : {v}")
    logger.info("─" * 60)
