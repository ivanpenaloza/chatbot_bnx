"""Base protocol for document text extractors.

Compatible with Python 3.9.20.
To add a new extractor:
  1. Create a new file in this package implementing ExtractorProtocol.
  2. Register the file extension(s) in __init__.py EXTRACTORS dict.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ExtractorProtocol(Protocol):
    """Protocol every extractor must satisfy."""

    def extract(self, filepath: str) -> str:
        """Extract plain text from the file at *filepath*.

        Returns the extracted text as a UTF-8 string.
        Raises ValueError for unsupported or unreadable files.
        """
        ...
