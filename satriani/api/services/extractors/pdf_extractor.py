"""Extractor for .pdf files using pymupdf (fitz).

Compatible with Python 3.9.20.
"""

import re


def _fix_pdf_spacing(text: str) -> str:
    """Re-insert spaces into text where PDF extraction concatenated words."""
    if not text:
        return text
    # lowercase followed by uppercase: camelCase word boundaries
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # letter followed by digit
    text = re.sub(r'([a-zA-Z]{2,})(\d)', r'\1 \2', text)
    # digit followed by letter (keep ordinals like 2nd, 3rd)
    text = re.sub(
        r'(\d)([a-zA-Z])',
        lambda m: m.group(0) if m.group(2) in ('st', 'nd', 'rd', 'th') else m.group(1) + ' ' + m.group(2),
        text,
    )
    # punctuation followed by letter with no space
    text = re.sub(r'([.,;:])([A-Za-z])', r'\1 \2', text)
    # closing paren followed by letter
    text = re.sub(r'\)([A-Za-z])', r') \1', text)
    # letter followed by opening paren
    text = re.sub(r'([a-zA-Z])\(', r'\1 (', text)
    return text


class PdfExtractor:
    """Extract text from PDF files using MuPDF (pymupdf / fitz)."""

    def extract(self, filepath: str) -> str:
        import fitz  # pymupdf

        text_parts = []
        with fitz.open(filepath) as doc:
            for page in doc:
                t = page.get_text("text")
                if t:
                    text_parts.append(t.strip())

        raw = "\n".join(text_parts)
        return _fix_pdf_spacing(raw)
