"""PDF exporter using fpdf2.

Converts Markdown text to a well-formatted A4 PDF with the Banamex logo
printed in the top-left corner of every page via the header() callback.

Pipeline:
  Markdown → HTML  (via `markdown` standard library)
  HTML     → PDF   (via fpdf2 write_html)

Compatible with Python 3.9.20.
"""

from __future__ import annotations

import os
import io
import html as html_module
from typing import ClassVar

import markdown as md_lib
from fpdf import FPDF
from fpdf.html import HTMLMixin

from .logo_utils import get_logo_path
# ── Unicode font (DejaVu Sans) ─────────────────────────────────────────────────────
_DEJAVU_DIR = "/usr/share/fonts/truetype/dejavu"
_FONT_FAMILY = "DejaVu"


def _register_fonts(pdf: FPDF) -> None:
    """Register DejaVu Sans (regular, bold, italic, bold-italic) so that
    write_html() can render Unicode characters including curly quotes,
    em-dashes, and other non-Latin-1 glyphs."""
    variants = [
        ("",   "DejaVuSans.ttf"),
        ("B",  "DejaVuSans-Bold.ttf"),
        ("I",  "DejaVuSans-Oblique.ttf"),
        ("BI", "DejaVuSans-BoldOblique.ttf"),
    ]
    for style, fname in variants:
        path = os.path.join(_DEJAVU_DIR, fname)
        if os.path.isfile(path):
            pdf.add_font(_FONT_FAMILY, style=style, fname=path)

# ── Layout constants ───────────────────────────────────────────────────────────
_LOGO_X_MM = 10.0        # logo left edge (mm from page left)
_LOGO_Y_MM = 8.0         # logo top edge  (mm from page top)
_LOGO_W_MM = 38.0        # logo width     (height is auto-scaled by fpdf2)
_HEADER_H_MM = 22.0      # total header band height (text top margin)
_MARGIN_MM = 20.0        # left / right / bottom page margin


class _BanamexPDF(FPDF, HTMLMixin):
    """FPDF subclass that paints the Banamex logo in the header of every page."""

    def header(self) -> None:  # type: ignore[override]
        # Draw a subtle bottom border for the header band
        self.set_line_width(0.3)
        self.set_draw_color(200, 200, 200)

        # Logo — only if the file exists
        logo = get_logo_path()
        if logo:
            self.image(logo, x=_LOGO_X_MM, y=_LOGO_Y_MM, w=_LOGO_W_MM)

        # Thin separator line below header
        y_line = _LOGO_Y_MM + _LOGO_W_MM * 0.35 + 2  # approximate logo height
        self.line(_MARGIN_MM, y_line, self.w - _MARGIN_MM, y_line)
        self.ln(2)

    def footer(self) -> None:  # type: ignore[override]
        self.set_y(-15)
        self.set_font(_FONT_FAMILY, "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _md_to_html(markdown_text: str) -> str:
    """Convert Markdown to HTML using the `markdown` package.

    Extensions used:
      - tables    — GitHub-style pipe tables
      - fenced_code — ```code``` blocks
      - nl2br     — single newline → <br>
    """
    return md_lib.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "nl2br"],
    )


class FPdf2Exporter:
    """Export Markdown content to PDF with Banamex logo on every page."""

    content_type: ClassVar[str] = "application/pdf"
    extension: ClassVar[str] = ".pdf"

    def export(self, content: str, title: str) -> bytes:
        # Normalise line endings
        content = content.replace("\\n", "\n")

        # Convert Markdown → HTML
        html_body = _md_to_html(content)

        # ── Build PDF ──────────────────────────────────────────────────
        pdf = _BanamexPDF(orientation="P", unit="mm", format="A4")
        _register_fonts(pdf)  # must happen before add_page()
        pdf.set_margins(_MARGIN_MM, _HEADER_H_MM, _MARGIN_MM)
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.add_page()

        # Default body font (Unicode-capable DejaVu Sans)
        pdf.set_font(_FONT_FAMILY, size=11)
        pdf.set_text_color(26, 26, 26)  # near-black

        # Render HTML body
        pdf.write_html(html_body)

        return bytes(pdf.output())
