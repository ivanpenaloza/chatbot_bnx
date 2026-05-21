"""Exporter registry for Satriani document generation.

To add a new export format:
  1. Create a new file in this package implementing ExporterProtocol.
  2. Add the format key → exporter class mapping to EXPORTERS below.

Compatible with Python 3.9.20.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import ExporterProtocol
from .pdf_exporter import FPdf2Exporter
from .docx_exporter import PythonDocxExporter

# ─── Registry ─────────────────────────────────────────────────────────────────
# Map format key (lowercase, no dot) → exporter class.
# To register a new format, add an entry here.

EXPORTERS: Dict[str, Type[ExporterProtocol]] = {
    "pdf":  FPdf2Exporter,
    "docx": PythonDocxExporter,
}

SUPPORTED_FORMATS = frozenset(EXPORTERS.keys())


def get_exporter(fmt: str) -> ExporterProtocol:
    """Return an instantiated exporter for *fmt*.

    Raises ValueError if the format is not registered.
    """
    fmt = fmt.lower().strip()
    exporter_cls = EXPORTERS.get(fmt)
    if exporter_cls is None:
        raise ValueError(
            f"Unsupported export format '{fmt}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    return exporter_cls()
