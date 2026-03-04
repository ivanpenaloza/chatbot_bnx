"""
Auth Router — Satriani Authentication

Simple JSON-file-based auth with session tokens.
Supports admin and regular user roles.
Compatible with Python 3.9.20.
"""

import os
import sys
import json
import hashlib
import secrets
import logging
from typing import Optional, Dict

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import (
    AUTH_DB_PATH,
    ADMIN_DEFAULT_USERNAME,
    ADMIN_DEFAULT_PASSWORD,
    SESSION_SECRET,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# ─── In-memory session store ─────────────────────────────────────────────────
_sessions: Dict[str, dict] = {}  # token -> {username, role}


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterUserRequest(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = ""


# ─── User DB helpers ─────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    return hashlib.sha256((password + SESSION_SECRET).encode()).hexdigest()


def _load_users() -> dict:
    """Load users from JSON file. Creates default admin if file missing."""
    if os.path.exists(AUTH_DB_PATH):
        try:
            with open(AUTH_DB_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    # Create default with admin
    users = {
        ADMIN_DEFAULT_USERNAME: {
            "password_hash": _hash_password(ADMIN_DEFAULT_PASSWORD),
            "role": "admin",
            "full_name": "Administrator",
        }
    }
    _save_users(users)
    return users


def _save_users(users: dict):
    os.makedirs(os.path.dirname(AUTH_DB_PATH), exist_ok=True)
    with open(AUTH_DB_PATH, 'w') as f:
        json.dump(users, f, indent=2)


# ─── Auth dependency ─────────────────────────────────────────────────────────

def get_current_user(request: Request) -> dict:
    """Extract user from Authorization header (Bearer token)."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth[7:]
    user = _sessions.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


def require_admin(request: Request) -> dict:
    """Require admin role."""
    user = get_current_user(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("/login")
async def login(req: LoginRequest):
    """Authenticate user or admin and return session token."""
    users = _load_users()
    user = users.get(req.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if user["password_hash"] != _hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = secrets.token_hex(32)
    _sessions[token] = {
        "username": req.username,
        "role": user["role"],
        "full_name": user.get("full_name", ""),
    }
    return JSONResponse(content={
        "token": token,
        "username": req.username,
        "role": user["role"],
        "full_name": user.get("full_name", ""),
    })


@router.post("/logout")
async def logout(request: Request):
    """Invalidate session."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        _sessions.pop(token, None)
    return JSONResponse(content={"status": "ok"})


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Return current user info."""
    return JSONResponse(content=user)


@router.post("/register")
async def register_user(req: RegisterUserRequest, admin: dict = Depends(require_admin)):
    """Admin-only: register a new user."""
    users = _load_users()
    if req.username in users:
        raise HTTPException(status_code=409, detail="Username already exists")
    if len(req.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")

    users[req.username] = {
        "password_hash": _hash_password(req.password),
        "role": "user",
        "full_name": req.full_name or req.username,
    }
    _save_users(users)
    return JSONResponse(content={"status": "created", "username": req.username})


@router.get("/users")
async def list_users(admin: dict = Depends(require_admin)):
    """Admin-only: list all registered users."""
    users = _load_users()
    result = []
    for uname, info in users.items():
        result.append({
            "username": uname,
            "role": info["role"],
            "full_name": info.get("full_name", ""),
        })
    return JSONResponse(content={"users": result})


@router.delete("/users/{username}")
async def delete_user(username: str, admin: dict = Depends(require_admin)):
    """Admin-only: delete a user (cannot delete admin)."""
    if username == ADMIN_DEFAULT_USERNAME:
        raise HTTPException(status_code=400, detail="Cannot delete the default admin")
    users = _load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    del users[username]
    _save_users(users)
    return JSONResponse(content={"status": "deleted", "username": username})
