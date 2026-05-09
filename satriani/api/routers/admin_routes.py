"""
Admin Router — Satriani Admin Panel

Provides admin-only endpoints for:
  - Serving the admin panel HTML
  - Uploading / deleting documents
  - Listing embedding models
  - Deleting ChromaDB collections (embeddings)

Compatible with Python 3.9.20.
"""

import os
import sys
import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import (
    RAG_DATA_DIR,
    RAG_UPLOAD_MAX_SIZE_MB,
    AVAILABLE_EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_KEY,
    CHROMA_PERSIST_DIR,
)
from routers.auth_routes import get_current_user, require_admin
import db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
templates = Jinja2Templates(directory="templates")


@router.get("/panel", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Serve the admin panel page."""
    return templates.TemplateResponse("admin.html", {"request": request})


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    admin: dict = Depends(require_admin),
):
    """Upload a docx or pdf file to the RAG data directory."""
    fname = file.filename or "unknown"
    ext = os.path.splitext(fname)[1].lower()
    if ext not in (".docx", ".pdf"):
        raise HTTPException(status_code=400, detail="Only .docx and .pdf files are allowed")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > RAG_UPLOAD_MAX_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Max is {RAG_UPLOAD_MAX_SIZE_MB} MB."
        )

    os.makedirs(RAG_DATA_DIR, exist_ok=True)
    dest = os.path.join(RAG_DATA_DIR, fname)

    # Prevent uploading a file that already exists on disk
    if os.path.exists(dest):
        raise HTTPException(
            status_code=409,
            detail=f"Document '{fname}' already exists. Delete it first if you want to re-upload."
        )

    with open(dest, "wb") as f:
        f.write(content)

    size_kb = round(len(content) / 1024, 1)
    db.register_document(fname, size_kb, admin.get("username", "admin"))

    logger.info("Uploaded document: %s (%.1f KB)", fname, size_kb)
    return JSONResponse(content={"status": "uploaded", "filename": fname, "size_kb": size_kb})


@router.delete("/documents/{filename}")
async def delete_document(filename: str, admin: dict = Depends(require_admin)):
    """Delete a document file from disk. Does NOT delete its ChromaDB embedding."""
    fpath = os.path.join(RAG_DATA_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(fpath)
    db.unregister_document(filename)
    logger.info("Deleted document: %s", filename)
    return JSONResponse(content={"status": "deleted", "filename": filename})


@router.get("/embedding-models")
async def list_embedding_models(user: dict = Depends(get_current_user)):
    """List available embedding models (any authenticated user)."""
    models = []
    for key, info in AVAILABLE_EMBEDDING_MODELS.items():
        models.append({
            "key": key,
            "display_name": info["display_name"],
            "context_length": info["context_length"],
            "description": info["description"],
            "available": os.path.isdir(info["local_dir"]),
            "active": key == DEFAULT_EMBEDDING_KEY,
        })
    return JSONResponse(content={"models": models, "active": DEFAULT_EMBEDDING_KEY})


@router.get("/embeddings")
async def list_embeddings(user: dict = Depends(get_current_user)):
    """List all ChromaDB collections (embeddings) with metadata."""
    from routers.rag_routes import get_chroma_client
    client = get_chroma_client()
    collections = client.list_collections()
    result = []
    for col in collections:
        meta = col.metadata or {}
        result.append({
            "collection_name": col.name,
            "source_file": meta.get("source_file", "unknown"),
            "embedding_model": meta.get("embedding_model", "unknown"),
            "chunks": col.count(),
        })
    return JSONResponse(content={"embeddings": result})


@router.delete("/embeddings/{collection_name}")
async def delete_embedding(collection_name: str, admin: dict = Depends(require_admin)):
    """Delete a ChromaDB collection and its on-disk HNSW segment folders."""
    import shutil
    from routers.rag_routes import get_chroma_client
    client = get_chroma_client()
    try:
        client.get_collection(name=collection_name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection not found: {collection_name}")

    # Snapshot UUID folders before deletion
    before = set()
    if os.path.isdir(CHROMA_PERSIST_DIR):
        before = {d for d in os.listdir(CHROMA_PERSIST_DIR) if os.path.isdir(os.path.join(CHROMA_PERSIST_DIR, d))}

    client.delete_collection(name=collection_name)

    # Snapshot after — any folder that's gone from ChromaDB's tracking is orphaned
    after = set()
    if os.path.isdir(CHROMA_PERSIST_DIR):
        after = {d for d in os.listdir(CHROMA_PERSIST_DIR) if os.path.isdir(os.path.join(CHROMA_PERSIST_DIR, d))}

    # Remove orphaned UUID segment folders
    orphaned = before - after
    # ChromaDB may not remove them from the filesystem itself, so also check
    # for UUID-shaped dirs that no longer map to any collection
    remaining_ids = set()
    for col in client.list_collections():
        # ChromaDB internally tracks segment UUIDs; we compare what's on disk
        pass
    # Clean up any dirs that were present before but are no longer needed
    for dirname in before:
        dirpath = os.path.join(CHROMA_PERSIST_DIR, dirname)
        if os.path.isdir(dirpath) and _is_uuid(dirname):
            # Check if this UUID dir is still referenced by any remaining collection
            if not _uuid_dir_is_referenced(CHROMA_PERSIST_DIR, dirname):
                shutil.rmtree(dirpath, ignore_errors=True)
                logger.info("Removed orphaned segment folder: %s", dirname)

    logger.info("Deleted ChromaDB collection: %s", collection_name)
    return JSONResponse(content={"status": "deleted", "collection": collection_name})


def _is_uuid(name: str) -> bool:
    """Check if a directory name looks like a UUID."""
    import re
    return bool(re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', name))


def _uuid_dir_is_referenced(chroma_dir: str, uuid_name: str) -> bool:
    """Check ChromaDB's SQLite to see if a UUID segment is still referenced."""
    import sqlite3
    db_path = os.path.join(chroma_dir, "chroma.sqlite3")
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        # ChromaDB stores segment IDs in the 'segments' table
        cur = conn.execute("SELECT 1 FROM segments WHERE id=? LIMIT 1", (uuid_name,))
        result = cur.fetchone() is not None
        conn.close()
        return result
    except Exception:
        return True  # If we can't check, don't delete to be safe
