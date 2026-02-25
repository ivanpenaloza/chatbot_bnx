"""
Base module for question-specific routers.

Provides shared utilities:
- SQLite-based querying of cubo_datos_v2.csv (loaded once)
- LLM response generation with context injection
- Common data structures

Compatible with Python 3.9.20.
"""

import os
import sys
import sqlite3
import time
import logging
import traceback
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import CHATBOT_CSV_PATH

logger = logging.getLogger(__name__)

# ─── Singleton: in-memory SQLite with CSV data ──────────────────────────────

_sqlite_conn = None  # type: Optional[sqlite3.Connection]
_df_cache = None  # type: Optional[pd.DataFrame]


def _get_csv_full_path() -> str:
    full = os.path.join(os.path.dirname(os.path.dirname(__file__)), CHATBOT_CSV_PATH)
    if os.path.exists(full):
        return full
    if os.path.exists(CHATBOT_CSV_PATH):
        return CHATBOT_CSV_PATH
    raise FileNotFoundError(f"CSV not found: {full} or {CHATBOT_CSV_PATH}")


def get_dataframe() -> pd.DataFrame:
    """Return cached DataFrame of cubo_datos_v2."""
    global _df_cache
    if _df_cache is None:
        path = _get_csv_full_path()
        _df_cache = pd.read_csv(path, low_memory=False)
        logger.info("DataFrame loaded: %d rows", len(_df_cache))
    return _df_cache


def get_sqlite_conn() -> sqlite3.Connection:
    """Return a shared in-memory SQLite connection with the CSV loaded as table 'cubo'."""
    global _sqlite_conn
    if _sqlite_conn is None:
        df = get_dataframe()
        _sqlite_conn = sqlite3.connect(":memory:", check_same_thread=False)
        df.to_sql("cubo", _sqlite_conn, index=False, if_exists="replace")
        logger.info("SQLite in-memory table 'cubo' created with %d rows", len(df))
    return _sqlite_conn


def run_query(sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    """Execute a SQL query against the cubo table and return list of dicts."""
    conn = get_sqlite_conn()
    try:
        cursor = conn.execute(sql, params or ())
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.error("SQL error: %s\nQuery: %s", e, sql)
        return []


def run_query_as_text(sql: str, params: Optional[tuple] = None) -> str:
    """Execute SQL and return a formatted text table for LLM context."""
    results = run_query(sql, params)
    if not results:
        return "(no results)"
    keys = list(results[0].keys())
    lines = [" | ".join(keys)]
    lines.append("-" * len(lines[0]))
    for row in results:
        vals = []
        for k in keys:
            v = row[k]
            if isinstance(v, float):
                if abs(v) >= 1_000_000:
                    vals.append(f"{v:,.0f}")
                else:
                    vals.append(f"{v:,.4f}")
            else:
                vals.append(str(v) if v is not None else "N/A")
        lines.append(" | ".join(vals))
    return "\n".join(lines)


def get_categorical_values(column: str) -> List[str]:
    """Return sorted unique non-null values for a categorical column."""
    df = get_dataframe()
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().unique().tolist())
