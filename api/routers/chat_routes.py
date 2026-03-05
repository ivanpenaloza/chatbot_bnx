"""
Chat Router — Satriani Chat Management

Provides REST API endpoints for managing per-user chats and messages.
Each user has their own chat history stored server-side in SQLite.

Compatible with Python 3.9.20.
"""

import os
import sys
import json
import logging
import secrets
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from routers.auth_routes import get_current_user
import db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chats", tags=["chats"])

MAX_CHATS_PER_USER = 50


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class CreateChatRequest(BaseModel):
    title: Optional[str] = "New Chat"


class UpdateChatRequest(BaseModel):
    title: str


class AddMessageRequest(BaseModel):
    role: str
    content: str
    rag_sources: Optional[List] = []
    uploaded_sources: Optional[List] = []
    inference_time: Optional[float] = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("")
async def list_user_chats(user: dict = Depends(get_current_user)):
    """List all chats for the current user."""
    chats = db.list_chats(user["username"])
    return JSONResponse(content={"chats": chats})


@router.post("")
async def create_chat(req: CreateChatRequest, user: dict = Depends(get_current_user)):
    """Create a new chat for the current user."""
    existing = db.list_chats(user["username"])
    if len(existing) >= MAX_CHATS_PER_USER:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_CHATS_PER_USER} chats reached.")
    chat_id = "chat_" + secrets.token_hex(12)
    chat = db.create_chat(chat_id, user["username"], req.title or "New Chat")
    return JSONResponse(content={"chat": chat})


@router.get("/{chat_id}")
async def get_chat(chat_id: str, user: dict = Depends(get_current_user)):
    """Get a chat with all its messages."""
    chat = db.get_chat(chat_id, user["username"])
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    messages = db.get_messages(chat_id)
    chat["messages"] = messages
    return JSONResponse(content={"chat": chat})


@router.patch("/{chat_id}")
async def update_chat(chat_id: str, req: UpdateChatRequest, user: dict = Depends(get_current_user)):
    """Update a chat's title."""
    ok = db.update_chat_title(chat_id, user["username"], req.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Chat not found")
    return JSONResponse(content={"status": "updated"})


@router.delete("/{chat_id}")
async def delete_chat(chat_id: str, user: dict = Depends(get_current_user)):
    """Delete a chat and all its messages."""
    ok = db.delete_chat(chat_id, user["username"])
    if not ok:
        raise HTTPException(status_code=404, detail="Chat not found")
    return JSONResponse(content={"status": "deleted"})


@router.post("/{chat_id}/messages")
async def add_message(chat_id: str, req: AddMessageRequest, user: dict = Depends(get_current_user)):
    """Add a message to a chat."""
    chat = db.get_chat(chat_id, user["username"])
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    if req.role not in ("user", "bot"):
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'bot'")
    msg = db.add_message(
        chat_id=chat_id,
        role=req.role,
        content=req.content,
        rag_sources=json.dumps(req.rag_sources or []),
        uploaded_sources=json.dumps(req.uploaded_sources or []),
        inference_time=req.inference_time,
    )
    return JSONResponse(content={"message": msg})


@router.get("/{chat_id}/messages")
async def get_messages(chat_id: str, user: dict = Depends(get_current_user)):
    """Get all messages for a chat."""
    chat = db.get_chat(chat_id, user["username"])
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    messages = db.get_messages(chat_id)
    return JSONResponse(content={"messages": messages})
