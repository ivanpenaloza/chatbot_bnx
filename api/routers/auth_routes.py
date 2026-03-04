"""
Auth Router — Satriani Authentication

SQLite-backed auth with persistent session tokens.
Supports admin and regular user roles.
Compatible with Python 3.9.20.
"""

import os
import sys
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import ADMIN_DEFAULT_USERNAME
import db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterUserRequest(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = ""


# ─── Auth Dependencies ───────────────────────────────────────────────────────

def get_current_user(request: Request) -> dict:
    """Extract user from Authorization header (Bearer token)."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth[7:]
    user = db.get_session(token)
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
    user = db.authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = db.create_session(user["username"], user["role"], user["full_name"])
    return JSONResponse(content={
        "token": token,
        "username": user["username"],
        "role": user["role"],
        "full_name": user["full_name"],
    })


@router.post("/logout")
async def logout(request: Request):
    """Invalidate session."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        db.delete_session(auth[7:])
    return JSONResponse(content={"status": "ok"})


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Return current user info."""
    return JSONResponse(content=user)


@router.post("/register")
async def register_user(req: RegisterUserRequest, admin: dict = Depends(require_admin)):
    """Admin-only: register a new user."""
    if len(req.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")

    ok = db.create_user(req.username, req.password, req.full_name or req.username)
    if not ok:
        raise HTTPException(status_code=409, detail="Username already exists")
    return JSONResponse(content={"status": "created", "username": req.username})


@router.get("/users")
async def list_users(admin: dict = Depends(require_admin)):
    """Admin-only: list all registered users."""
    return JSONResponse(content={"users": db.list_users()})


@router.delete("/users/{username}")
async def delete_user(username: str, admin: dict = Depends(require_admin)):
    """Admin-only: delete a user (cannot delete admin)."""
    if username == ADMIN_DEFAULT_USERNAME:
        raise HTTPException(status_code=400, detail="Cannot delete the default admin")
    ok = db.delete_user(username)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return JSONResponse(content={"status": "deleted", "username": username})
