"""
Database Layer — Satriani Platform

Single SQLite database for users, sessions, chats, messages, and document metadata.
ChromaDB keeps its own SQLite for embeddings — we reference collections
by name but don't duplicate vector data here.

Schema:
  users    1──N  chats    (a user owns many chats)
  chats    1──N  messages (a chat contains many messages)
  users    1──N  sessions
  documents (standalone)

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

    # Create tables only if they don't already exist (preserves data across restarts)
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
        CREATE TABLE IF NOT EXISTS chats (
            id          TEXT PRIMARY KEY,
            username    TEXT NOT NULL,
            title       TEXT NOT NULL DEFAULT 'New Chat',
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     TEXT NOT NULL,
            role        TEXT NOT NULL CHECK(role IN ('user', 'bot')),
            content     TEXT NOT NULL,
            rag_sources TEXT DEFAULT '[]',
            uploaded_sources TEXT DEFAULT '[]',
            inference_time REAL DEFAULT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_chats_username ON chats(username);
        CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
        CREATE TABLE IF NOT EXISTS documents (
            filename    TEXT PRIMARY KEY,
            size_kb     REAL NOT NULL DEFAULT 0,
            uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
            uploaded_by TEXT NOT NULL DEFAULT 'system'
        );
    """)
    conn.commit()

    # Seed default admin
    row = conn.execute("SELECT 1 FROM users WHERE username=?", (admin_user,)).fetchone()
    if not row:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, full_name) VALUES (?,?,?,?)",
            (admin_user, _hash_password(admin_pass), "admin", "Administrator"),
        )
        conn.commit()
        logger.info("Default admin user created.")

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
    conn.execute("DELETE FROM sessions WHERE username=?", (username,))
    cur = conn.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    return cur.rowcount > 0


# ─── Chat Operations ─────────────────────────────────────────────────────────

def create_chat(chat_id: str, username: str, title: str = "New Chat") -> Dict[str, Any]:
    """Create a new chat for a user."""
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO chats (id, username, title, created_at, updated_at) VALUES (?,?,?,?,?)",
        (chat_id, username, title, now, now),
    )
    conn.commit()
    return {"id": chat_id, "username": username, "title": title, "created_at": now, "updated_at": now}


def list_chats(username: str) -> List[Dict[str, Any]]:
    """List all chats for a user, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, username, title, created_at, updated_at FROM chats WHERE username=? ORDER BY updated_at DESC",
        (username,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_chat(chat_id: str, username: str) -> Optional[Dict[str, Any]]:
    """Get a single chat, verifying ownership."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, username, title, created_at, updated_at FROM chats WHERE id=? AND username=?",
        (chat_id, username),
    ).fetchone()
    return dict(row) if row else None


def update_chat_title(chat_id: str, username: str, title: str) -> bool:
    """Update a chat's title."""
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        "UPDATE chats SET title=?, updated_at=? WHERE id=? AND username=?",
        (title, now, chat_id, username),
    )
    conn.commit()
    return cur.rowcount > 0


def delete_chat(chat_id: str, username: str) -> bool:
    """Delete a chat and all its messages (CASCADE)."""
    conn = _get_conn()
    cur = conn.execute("DELETE FROM chats WHERE id=? AND username=?", (chat_id, username))
    conn.commit()
    return cur.rowcount > 0


# ─── Message Operations ──────────────────────────────────────────────────────

def add_message(chat_id: str, role: str, content: str,
                rag_sources: str = "[]", uploaded_sources: str = "[]",
                inference_time: Optional[float] = None) -> Dict[str, Any]:
    """Add a message to a chat. Also updates chat.updated_at and auto-titles."""
    import json
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        "INSERT INTO messages (chat_id, role, content, rag_sources, uploaded_sources, inference_time, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (chat_id, role, content, rag_sources, uploaded_sources, inference_time, now),
    )
    conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (now, chat_id))

    # Auto-title: set chat title from first user message
    if role == "user":
        chat_row = conn.execute("SELECT title FROM chats WHERE id=?", (chat_id,)).fetchone()
        if chat_row and chat_row["title"] == "New Chat":
            auto_title = content[:50] + ("..." if len(content) > 50 else "")
            conn.execute("UPDATE chats SET title=? WHERE id=?", (auto_title, chat_id))

    conn.commit()
    return {
        "id": cur.lastrowid,
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "rag_sources": json.loads(rag_sources),
        "uploaded_sources": json.loads(uploaded_sources),
        "inference_time": inference_time,
        "created_at": now,
    }


def get_messages(chat_id: str) -> List[Dict[str, Any]]:
    """Get all messages for a chat, ordered chronologically."""
    import json
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, chat_id, role, content, rag_sources, uploaded_sources, inference_time, created_at "
        "FROM messages WHERE chat_id=? ORDER BY id ASC",
        (chat_id,),
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["rag_sources"] = json.loads(d["rag_sources"]) if d["rag_sources"] else []
        d["uploaded_sources"] = json.loads(d["uploaded_sources"]) if d["uploaded_sources"] else []
        result.append(d)
    return result


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
