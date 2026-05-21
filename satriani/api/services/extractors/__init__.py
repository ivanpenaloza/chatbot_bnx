"""Extractor registry for Satriani document uploads.

To add support for a new file type:
  1. Create a new extractor module in this package that implements ExtractorProtocol.
  2. Add the file extension(s) → extractor class mapping to EXTRACTORS below.

Compatible with Python 3.9.20.
"""

from __future__ import annotations

import os
from typing import Dict, Type

from .base import ExtractorProtocol
from .docx_extractor import DocxExtractor
from .pdf_extractor import PdfExtractor
from .excel_extractor import ExcelExtractor, CsvExtractor

# ─── Registry ─────────────────────────────────────────────────────────────────
# Map lowercase file extension (with leading dot) → extractor class.
# To register a new format, add an entry here.

EXTRACTORS: Dict[str, Type[ExtractorProtocol]] = {
    ".docx": DocxExtractor,
    ".pdf":  PdfExtractor,
    ".xlsx": ExcelExtractor,
    ".xls":  ExcelExtractor,
    ".csv":  CsvExtractor,
}

# Flat set of all supported extensions (for validation in routes)
SUPPORTED_EXTENSIONS = frozenset(EXTRACTORS.keys())


def extract_text(filepath: str) -> str:
    """Extract plain text from a file, dispatching to the correct extractor.

    Raises ValueError for unsupported file types.
    """
    ext = os.path.splitext(filepath)[1].lower()
    extractor_cls = EXTRACTORS.get(ext)
    if extractor_cls is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    extractor = extractor_cls()
    return extractor.extract(filepath)
