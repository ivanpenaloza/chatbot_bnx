"""
Admin Router — Satriani Admin Panel

Provides admin-only endpoints for:
  - Serving the admin panel HTML
  - Uploading documents (docx/pdf) to the RAG data directory
  - Triggering document ingestion
  - Viewing embedding model info

Compatible with Python 3.9.20.
"""

import os
import sys
import shutil
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
)
from routers.auth_routes import require_admin

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

    # Read and check size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > RAG_UPLOAD_MAX_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Max is {RAG_UPLOAD_MAX_SIZE_MB} MB."
        )

    os.makedirs(RAG_DATA_DIR, exist_ok=True)
    dest = os.path.join(RAG_DATA_DIR, fname)
    with open(dest, "wb") as f:
        f.write(content)

    logger.info("Uploaded document: %s (%.1f MB)", fname, size_mb)
    return JSONResponse(content={
        "status": "uploaded",
        "filename": fname,
        "size_mb": round(size_mb, 2),
    })


@router.delete("/documents/{filename}")
async def delete_document(filename: str, admin: dict = Depends(require_admin)):
    """Delete a document from the RAG data directory."""
    fpath = os.path.join(RAG_DATA_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(fpath)
    logger.info("Deleted document: %s", filename)
    return JSONResponse(content={"status": "deleted", "filename": filename})


@router.get("/embedding-models")
async def list_embedding_models(admin: dict = Depends(require_admin)):
    """List available embedding models."""
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
