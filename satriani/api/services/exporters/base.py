"""Base protocol for document exporters.

Compatible with Python 3.9.20.
To add a new export format:
  1. Create a new file in this package implementing ExporterProtocol.
  2. Register the format key in __init__.py EXPORTERS dict.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ExporterProtocol(Protocol):
    """Protocol every exporter must satisfy."""

    content_type: str   # MIME type returned to the browser
    extension: str      # File extension with leading dot, e.g. ".pdf"

    def export(self, content: str, title: str) -> bytes:
        """Convert *content* (Markdown text) to file bytes.

        Args:
            content: The Markdown string to export.
            title:   A suggested filename stem (no extension).

        Returns:
            Raw bytes of the generated file.
        """
        ...
