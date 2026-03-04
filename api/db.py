"""
Database Layer — Satriani Platform

Single SQLite database for users, sessions, and document metadata.
ChromaDB keeps its own SQLite for embeddings — we reference collections
by name but don't duplicate vector data here.

Compatible with Python 3.9.20.
"""

import os
import sqlite3
import hashlib
import secrets
import logging
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Thread-local storage for connections
_local = threading.local()
_db_path: Optional[str] = None
_session_secret: str = ""


def init_db(db_path: str, session_secret: str, admin_user: str, admin_pass: str):
    """Initialize the database: create tables and default admin."""
    global _db_path, _session_secret
    _db_path = db_path
    _session_secret = session_secret
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            username    TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role        TEXT NOT NULL DEFAULT 'user',
            full_name   TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token       TEXT PRIMARY KEY,
            username    TEXT NOT NULL,
            role        TEXT NOT NULL,
            full_name   TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS documents (
            filename    TEXT PRIMARY KEY,
            size_kb     REAL NOT NULL DEFAULT 0,
            uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
            uploaded_by TEXT NOT NULL DEFAULT 'system'
        );
    """)
    conn.commit()

    # Seed default admin if not exists
    row = conn.execute("SELECT 1 FROM users WHERE username=?", (admin_user,)).fetchone()
    if not row:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, full_name) VALUES (?,?,?,?)",
            (admin_user, _hash_password(admin_pass), "admin", "Administrator"),
        )
        conn.commit()
        logger.info("Default admin user created.")

    # Migrate existing users.json if present
    _migrate_json_users(db_path, admin_user)
    logger.info("Database initialized at: %s", db_path)


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(_db_path, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def _hash_password(password: str) -> str:
    return hashlib.sha256((password + _session_secret).encode()).hexdigest()


def _migrate_json_users(db_path: str, admin_user: str):
    """One-time migration from users.json → SQLite."""
    import json
    json_path = os.path.join(os.path.dirname(db_path), "users.json")
    if not os.path.exists(json_path):
        return
    try:
        with open(json_path, "r") as f:
            users = json.load(f)
        conn = _get_conn()
        for uname, info in users.items():
            existing = conn.execute("SELECT 1 FROM users WHERE username=?", (uname,)).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO users (username, password_hash, role, full_name) VALUES (?,?,?,?)",
                    (uname, info["password_hash"], info.get("role", "user"), info.get("full_name", "")),
                )
        conn.commit()
        # Rename old file so migration doesn't repeat
        os.rename(json_path, json_path + ".migrated")
        logger.info("Migrated users.json → SQLite (%d users)", len(users))
    except Exception as e:
        logger.warning("Could not migrate users.json: %s", e)


# ─── User Operations ─────────────────────────────────────────────────────────

def authenticate(username: str, password: str) -> Optional[Dict[str, str]]:
    """Verify credentials. Returns user dict or None."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if not row:
        return None
    if row["password_hash"] != _hash_password(password):
        return None
    return {"username": row["username"], "role": row["role"], "full_name": row["full_name"]}


def create_session(username: str, role: str, full_name: str) -> str:
    """Create a session token and store it."""
    token = secrets.token_hex(32)
    conn = _get_conn()
    conn.execute(
        "INSERT INTO sessions (token, username, role, full_name) VALUES (?,?,?,?)",
        (token, username, role, full_name),
    )
    conn.commit()
    return token


def get_session(token: str) -> Optional[Dict[str, str]]:
    """Look up a session by token."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM sessions WHERE token=?", (token,)).fetchone()
    if not row:
        return None
    return {"username": row["username"], "role": row["role"], "full_name": row["full_name"]}


def delete_session(token: str):
    conn = _get_conn()
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()


def list_users() -> List[Dict[str, str]]:
    conn = _get_conn()
    rows = conn.execute("SELECT username, role, full_name FROM users ORDER BY username").fetchall()
    return [dict(r) for r in rows]


def create_user(username: str, password: str, full_name: str = "") -> bool:
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, full_name) VALUES (?,?,?,?)",
            (username, _hash_password(password), "user", full_name or username),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def delete_user(username: str) -> bool:
    conn = _get_conn()
    # Also delete their sessions
    conn.execute("DELETE FROM sessions WHERE username=?", (username,))
    cur = conn.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    return cur.rowcount > 0


# ─── Document Metadata Operations ────────────────────────────────────────────

def register_document(filename: str, size_kb: float, uploaded_by: str = "system"):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO documents (filename, size_kb, uploaded_by) VALUES (?,?,?)",
        (filename, size_kb, uploaded_by),
    )
    conn.commit()


def unregister_document(filename: str):
    conn = _get_conn()
    conn.execute("DELETE FROM documents WHERE filename=?", (filename,))
    conn.commit()


def list_documents() -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_document(filename: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM documents WHERE filename=?", (filename,)).fetchone()
    return dict(row) if row else None
