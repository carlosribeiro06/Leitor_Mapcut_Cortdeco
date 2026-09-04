"""Leitor dos binarios mapcut/cortdeco do DECOMP com exportacao para CSV."""

from __future__ import annotations

from .config import ConfigError, Settings, load_settings
from .geometry import CutGeometry, GeometryError
from .logging_setup import get_logger, setup_logging
from .pipeline import RunReport, run

__all__ = [
    "ConfigError",
    "CutGeometry",
    "GeometryError",
    "RunReport",
    "Settings",
    "get_logger",
    "load_settings",
    "run",
    "setup_logging",
]

__version__ = "1.0.0"
