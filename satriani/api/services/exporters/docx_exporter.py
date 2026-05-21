"""DOCX exporter using python-docx.

Converts Markdown text to a Word (.docx) document with the Banamex logo
in the section header (which repeats on every page automatically).

Pipeline:
  Markdown → HTML             (via `markdown` standard library)
  HTML     → intermediate AST (list of element dicts via html.parser)
  AST      → python-docx DOM  (two-pass: collect then render)

The two-pass approach avoids streaming parser problems with tables
(column count must be known before creating the DOCX table object).

Compatible with Python 3.9.20.
"""

from __future__ import annotations

import io
import os
import re
from html.parser import HTMLParser
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import markdown as md_lib
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: F401

from .logo_utils import get_logo_path

_LOGO_INCHES = 1.6


# ─── Phase 1: HTML → intermediate element list ───────────────────────────────
#
# Each element is a dict:
#   {"type": "heading",   "level": 1..6,  "runs": [...]}
#   {"type": "paragraph", "blockquote": bool, "runs": [...]}
#   {"type": "list_item", "ordered": bool, "runs": [...]}
#   {"type": "table",     "rows": [[cell_text, ...], ...], "has_header": True}
#   {"type": "pre",       "text": str}
#   {"type": "hr"}
#   {"type": "br"}
#
# Each run is a dict:
#   {"text": str, "bold": bool, "italic": bool, "code": bool}

def _mk_run(text: str, bold=False, italic=False, code=False) -> Dict[str, Any]:
    return {"text": text, "bold": bold, "italic": italic, "code": code}


class _HtmlCollector(HTMLParser):
    """Parse HTML and build a flat list of element dicts."""

    def __init__(self):
        super().__init__()
        self.elements: List[Dict[str, Any]] = []

        # Current state
        self._tag_stack: List[str] = []
        self._runs: List[Dict[str, Any]] = []
        self._bold = 0
        self._italic = 0
        self._code = 0
        self._current_type: Optional[str] = None  # heading, para, li, td…
        self._current_level = 0
        self._ordered = False
        self._blockquote = 0

        # Table buffering
        self._in_table = False
        self._table_rows: List[List[str]] = []
        self._table_has_header = False
        self._current_row: Optional[List[str]] = None
        self._current_cell_parts: List[str] = []
        self._in_th = False

        # Pre buffering
        self._in_pre = False
        self._pre_buf: List[str] = []

    # ── helpers ──────────────────────────────────────────────────────────

    def _flush_element(self):
        """Commit the current element (if any) and reset run state."""
        if self._current_type is None:
            self._runs = []
            return
        t = self._current_type
        if t == "heading":
            self.elements.append({"type": "heading", "level": self._current_level, "runs": self._runs})
        elif t == "paragraph":
            if any(r["text"].strip() for r in self._runs):
                self.elements.append({"type": "paragraph", "blockquote": self._blockquote > 0, "runs": self._runs})
        elif t == "list_item":
            self.elements.append({"type": "list_item", "ordered": self._ordered, "runs": self._runs})
        self._runs = []
        self._current_type = None
        self._current_level = 0

    def _push_run(self, text: str):
        if self._in_table:
            self._current_cell_parts.append(text)
            return
        if not text:
            return
        self._runs.append(_mk_run(
            text,
            bold=self._bold > 0,
            italic=self._italic > 0,
            code=self._code > 0,
        ))

    # ── HTMLParser callbacks ──────────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        tag = tag.lower()
        self._tag_stack.append(tag)

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush_element()
            self._current_type = "heading"
            self._current_level = int(tag[1])

        elif tag == "p":
            self._flush_element()
            self._current_type = "paragraph"

        elif tag == "li":
            self._flush_element()
            self._current_type = "list_item"

        elif tag == "ul":
            self._ordered = False

        elif tag == "ol":
            self._ordered = True

        elif tag == "table":
            self._flush_element()
            self._in_table = True
            self._table_rows = []
            self._table_has_header = False

        elif tag == "thead":
            self._table_has_header = True

        elif tag in ("tr",):
            self._current_row = []
            self._current_cell_parts = []

        elif tag in ("th", "td"):
            self._current_cell_parts = []
            self._in_th = (tag == "th")

        elif tag == "pre":
            self._flush_element()
            self._in_pre = True
            self._pre_buf = []

        elif tag == "code":
            if not self._in_pre:
                self._code += 1

        elif tag in ("strong", "b"):
            self._bold += 1

        elif tag in ("em", "i"):
            self._italic += 1

        elif tag == "blockquote":
            self._flush_element()
            self._blockquote += 1
            self._current_type = "paragraph"

        elif tag == "hr":
            self._flush_element()
            self.elements.append({"type": "hr"})

        elif tag == "br":
            self._push_run("\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li"):
            self._flush_element()

        elif tag in ("ul", "ol"):
            pass  # lists are implicit

        elif tag in ("th", "td"):
            if self._current_row is not None:
                self._current_row.append("".join(self._current_cell_parts))
            self._current_cell_parts = []
            self._in_th = False

        elif tag == "tr":
            if self._current_row is not None and self._current_row:
                self._table_rows.append(self._current_row)
            self._current_row = None

        elif tag == "table":
            self._in_table = False
            if self._table_rows:
                self.elements.append({
                    "type": "table",
                    "rows": self._table_rows,
                    "has_header": self._table_has_header,
                })
            self._table_rows = []

        elif tag == "pre":
            self._in_pre = False
            self.elements.append({"type": "pre", "text": "".join(self._pre_buf)})

        elif tag == "code":
            if self._code > 0:
                self._code -= 1

        elif tag in ("strong", "b"):
            if self._bold > 0:
                self._bold -= 1

        elif tag in ("em", "i"):
            if self._italic > 0:
                self._italic -= 1

        elif tag == "blockquote":
            self._flush_element()
            if self._blockquote > 0:
                self._blockquote -= 1

    def handle_data(self, data: str):
        if self._in_pre:
            self._pre_buf.append(data)
            return
        if self._in_table:
            self._current_cell_parts.append(data)
            return
        # Collapse whitespace outside pre/table
        text = re.sub(r'[ \t\r\n]+', ' ', data)
        if text:
            self._push_run(text)


# ─── Phase 2: element list → python-docx DOM ────────────────────────────────

def _apply_runs(para, runs: List[Dict[str, Any]]):
    """Add styled runs to a python-docx paragraph."""
    for r in runs:
        run = para.add_run(r["text"])
        run.bold = r.get("bold", False)
        run.italic = r.get("italic", False)
        if r.get("code"):
            run.font.name = "Courier New"
            run.font.size = Pt(9)


def _render_elements(doc: Document, elements: List[Dict[str, Any]]):
    for el in elements:
        kind = el["type"]

        if kind == "heading":
            p = doc.add_heading(level=el["level"])
            _apply_runs(p, el["runs"])

        elif kind == "paragraph":
            style = "Quote" if el.get("blockquote") else "Normal"
            p = doc.add_paragraph(style=style)
            _apply_runs(p, el["runs"])

        elif kind == "list_item":
            style = "List Number" if el.get("ordered") else "List Bullet"
            p = doc.add_paragraph(style=style)
            _apply_runs(p, el["runs"])

        elif kind == "table":
            rows = el["rows"]
            if not rows:
                continue
            n_cols = max(len(row) for row in rows)
            if n_cols == 0:
                continue
            tbl = doc.add_table(rows=len(rows), cols=n_cols)
            tbl.style = "Table Grid"
            for r_idx, row_cells in enumerate(rows):
                is_header = el.get("has_header") and r_idx == 0
                for c_idx in range(n_cols):
                    cell_text = row_cells[c_idx] if c_idx < len(row_cells) else ""
                    cell = tbl.cell(r_idx, c_idx)
                    p = cell.paragraphs[0]
                    run = p.add_run(cell_text)
                    if is_header:
                        run.bold = True

        elif kind == "pre":
            p = doc.add_paragraph(style="No Spacing")
            run = p.add_run(el["text"])
            run.font.name = "Courier New"
            run.font.size = Pt(9)
            # Light shading via XML
            from docx.oxml.ns import qn
            from lxml import etree
            pPr = p._element.get_or_add_pPr()
            shd = etree.SubElement(pPr, qn('w:shd'))
            shd.set(qn('w:val'), 'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'), 'F5F5F5')

        elif kind == "hr":
            p = doc.add_paragraph()
            from docx.oxml.ns import qn
            from lxml import etree
            pPr = p._element.get_or_add_pPr()
            pBdr = etree.SubElement(pPr, qn('w:pBdr'))
            bottom = etree.SubElement(pBdr, qn('w:bottom'))
            bottom.set(qn('w:val'), 'single')
            bottom.set(qn('w:sz'), '6')
            bottom.set(qn('w:space'), '1')
            bottom.set(qn('w:color'), 'CCCCCC')


def _md_to_html(markdown_text: str) -> str:
    return md_lib.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "nl2br"],
    )


def _remove_picture_border(run) -> None:
    """Remove the default black outline that Word adds around inline pictures."""
    for pic in run._r.iter(qn('pic:pic')):
        spPr = pic.find(qn('pic:spPr'))
        if spPr is None:
            continue
        # Drop any existing <a:ln> elements
        for ln in list(spPr.findall(qn('a:ln'))):
            spPr.remove(ln)
        # Add explicit no-border: <a:ln w="0"><a:noFill/></a:ln>
        ln_el = OxmlElement('a:ln')
        ln_el.set('w', '0')
        ln_el.append(OxmlElement('a:noFill'))
        spPr.append(ln_el)


def _add_logo_to_header(doc: Document) -> None:
    """Place the Banamex logo in the document's primary section header."""
    section = doc.sections[0]
    section.different_first_page_header_footer = False
    header = section.header
    header.is_linked_to_previous = False

    for para in header.paragraphs:
        para.clear()

    para = header.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    # The header paragraph starts at the text-area left edge (= page left margin).
    # Use a negative indent to push the logo towards the physical page edge,
    # matching the PDF position (~1 cm from the left side of the paper).
    para.paragraph_format.left_indent = Cm(-1.5)
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(0)

    logo = get_logo_path()
    if logo:
        run = para.add_run()
        run.add_picture(logo, width=Inches(_LOGO_INCHES))
        _remove_picture_border(run)
    else:
        run = para.add_run("Satriani — AI Document Intelligence")
        run.bold = True


class PythonDocxExporter:
    """Export Markdown content to DOCX with Banamex logo on every page."""

    content_type: ClassVar[str] = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    extension: ClassVar[str] = ".docx"

    def export(self, content: str, title: str) -> bytes:
        content = content.replace("\\n", "\n")

        doc = Document()

        # ── Page setup (A4, 2.5 cm margins) ──────────────────────────────
        section = doc.sections[0]
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

        # ── Logo header ───────────────────────────────────────────────────
        _add_logo_to_header(doc)

        # ── Parse and render body ─────────────────────────────────────────
        html_body = _md_to_html(content)
        collector = _HtmlCollector()
        collector.feed(html_body)
        # Flush any remaining open element
        collector._flush_element()

        _render_elements(doc, collector.elements)

        # ── Serialise ─────────────────────────────────────────────────────
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()


