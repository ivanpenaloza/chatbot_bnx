"""Extractor for Excel (.xlsx, .xls) and CSV files.

Converts all sheets to a structured markdown representation that is
optimised for small LLMs (Gemma 3 1B, Llama 3.1 8B).  Each sheet gets:
  - A header block with row/column counts and column type information
  - The data as a markdown table (≤200 rows) or key-value pairs for
    very wide sheets (>50 columns)
  - A brief numeric statistics section when numeric columns are present

The preamble explicitly tells the LLM what kind of data it is receiving,
which dramatically improves quality for small models that otherwise
struggle with tabular input.

Compatible with Python 3.9.20.
"""

from __future__ import annotations

import os
import io
import math
from typing import List, Optional

import pandas as pd

_MAX_ROWS_PER_SHEET = 200
_MAX_COLS_WIDE_THRESHOLD = 50
_MAX_CELL_LEN = 120          # truncate very long cell values
_MAX_TOTAL_CHARS = 60_000    # hard cap on total output to avoid token overflow


def _truncate_cell(value) -> str:
    """Convert a cell value to a clean string, truncating if necessary."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if len(text) > _MAX_CELL_LEN:
        text = text[:_MAX_CELL_LEN] + "…"
    return text


def _col_type_summary(series: pd.Series) -> str:
    """Return a short type+null summary for a single column."""
    dtype = str(series.dtype)
    null_count = int(series.isna().sum())
    if "int" in dtype or "float" in dtype:
        kind = "numeric"
    elif "datetime" in dtype or "date" in dtype:
        kind = "date"
    elif "bool" in dtype:
        kind = "boolean"
    else:
        kind = "text"
    if null_count:
        return f"{kind}, {null_count} null{'s' if null_count > 1 else ''}"
    return kind


def _numeric_stats(df: pd.DataFrame) -> Optional[str]:
    """Return a compact statistics block for numeric columns, or None."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        return None
    lines = ["**Numeric statistics:**"]
    for col in num_cols[:20]:  # cap at 20 numeric columns
        s = df[col].dropna()
        if s.empty:
            continue
        lines.append(
            f"- `{col}`: min={s.min():.4g}, max={s.max():.4g}, "
            f"mean={s.mean():.4g}, std={s.std():.4g}"
        )
    return "\n".join(lines) if len(lines) > 1 else None


def _sheet_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    df = df.fillna("")
    cols = df.columns.tolist()
    # Header row
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = " | ".join(_truncate_cell(row[c]) for c in cols)
        rows.append("| " + cells + " |")
    return "\n".join([header, separator] + rows)


def _sheet_to_key_value(df: pd.DataFrame) -> str:
    """Render a wide DataFrame as a list of record blocks (key: value pairs)."""
    df = df.fillna("")
    blocks = []
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        parts = [f"**Record {idx}:**"]
        for col in df.columns:
            val = _truncate_cell(row[col])
            if val:
                parts.append(f"  - {col}: {val}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def _process_sheet(sheet_name: str, df: pd.DataFrame, filename: str) -> str:
    """Convert a single sheet to its markdown representation."""
    nrows, ncols = df.shape

    # ── Column metadata ──────────────────────────────────────────────────
    col_info = ", ".join(
        f"`{col}` ({_col_type_summary(df[col])})" for col in df.columns[:30]
    )
    if ncols > 30:
        col_info += f", … and {ncols - 30} more columns"

    header_lines = [
        f"## Sheet: {sheet_name}  ({nrows} rows × {ncols} columns)",
        f"Columns: {col_info}",
    ]

    # ── Truncate to MAX_ROWS_PER_SHEET ───────────────────────────────────
    truncated = False
    if nrows > _MAX_ROWS_PER_SHEET:
        df = df.head(_MAX_ROWS_PER_SHEET)
        truncated = True
        header_lines.append(
            f"*(showing first {_MAX_ROWS_PER_SHEET} of {nrows} rows)*"
        )

    header_lines.append("")  # blank line before data

    # ── Data representation ──────────────────────────────────────────────
    if ncols > _MAX_COLS_WIDE_THRESHOLD:
        data_block = (
            "*Wide sheet — showing as records:*\n\n"
            + _sheet_to_key_value(df)
        )
    else:
        data_block = _sheet_to_markdown_table(df)

    # ── Numeric stats ────────────────────────────────────────────────────
    stats = _numeric_stats(df)
    parts = ["\n".join(header_lines), data_block]
    if stats:
        parts.append("\n" + stats)

    return "\n".join(parts)


def _read_excel_all_sheets(filepath: str) -> List[str]:
    """Read all sheets from an Excel file, return a list of sheet texts."""
    filename = os.path.basename(filepath)
    sheet_texts: List[str] = []

    xf = pd.ExcelFile(filepath, engine="openpyxl")
    for sheet_name in xf.sheet_names:
        try:
            df = pd.read_excel(xf, sheet_name=sheet_name, dtype=str)
            # Drop completely empty rows and columns
            df = df.dropna(how="all").dropna(axis=1, how="all")
            df.columns = [str(c).strip() for c in df.columns]
            if df.empty:
                continue
            # Re-read with inferred types for statistics only
            df_typed = pd.read_excel(xf, sheet_name=sheet_name)
            df_typed = df_typed.dropna(how="all").dropna(axis=1, how="all")
            df_typed.columns = [str(c).strip() for c in df_typed.columns]
            sheet_texts.append(_process_sheet(sheet_name, df, filename))
        except Exception as exc:
            sheet_texts.append(f"## Sheet: {sheet_name}\n*(Could not read: {exc})*")

    return sheet_texts


def _read_csv(filepath: str) -> List[str]:
    """Read a CSV file and return as a single-element list of sheet texts."""
    filename = os.path.basename(filepath)
    df = pd.read_csv(filepath, dtype=str)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        return ["*(CSV file is empty)*"]
    return [_process_sheet(filename, df, filename)]


class ExcelExtractor:
    """Extract structured text from Excel (.xlsx, .xls) files."""

    def extract(self, filepath: str) -> str:
        filename = os.path.basename(filepath)
        preamble = (
            f"=== STRUCTURED TABULAR DATA FROM: {filename} ===\n"
            "The following data is extracted from a spreadsheet.\n"
            "Analyze each sheet carefully. Preserve all names, dates, "
            "numbers, and decisions exactly as they appear.\n"
        )
        sheet_texts = _read_excel_all_sheets(filepath)
        if not sheet_texts:
            return preamble + "\n*(No readable sheets found)*"

        body = "\n\n---\n\n".join(sheet_texts)
        full = preamble + "\n" + body

        # Hard cap to protect context window
        if len(full) > _MAX_TOTAL_CHARS:
            full = full[:_MAX_TOTAL_CHARS] + "\n\n*(content truncated — file too large)*"
        return full


class CsvExtractor:
    """Extract structured text from CSV files."""

    def extract(self, filepath: str) -> str:
        filename = os.path.basename(filepath)
        preamble = (
            f"=== STRUCTURED TABULAR DATA FROM: {filename} ===\n"
            "The following data is extracted from a CSV file.\n"
        )
        sheet_texts = _read_csv(filepath)
        body = "\n\n".join(sheet_texts)
        full = preamble + "\n" + body

        if len(full) > _MAX_TOTAL_CHARS:
            full = full[:_MAX_TOTAL_CHARS] + "\n\n*(content truncated — file too large)*"
        return full
