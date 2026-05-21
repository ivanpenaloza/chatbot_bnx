"""Export Router — Satriani Document Generation

Provides server-side generation of PDF and DOCX files from Markdown content,
with the Banamex logo placed in the header of every page.

POST /api/v1/export/generate
  Body: { content: str, format: "pdf"|"docx", title?: str }
  Response: file bytes with appropriate Content-Type and Content-Disposition.

The exporter registry is defined in services/exporters/__init__.py.
To add a new format, register it there — no changes needed in this file.

Compatible with Python 3.9.20.
"""

import logging
import traceback
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from routers.auth_routes import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["export"])


# ─── Pydantic schema ──────────────────────────────────────────────────────────

class ExportRequest(BaseModel):
    content: str                  # Markdown text to export
    format: str                   # "pdf" or "docx"
    title: Optional[str] = "satriani-export"


# ─── Route ───────────────────────────────────────────────────────────────────

@router.post("/export/generate")
async def generate_export(
    req: ExportRequest,
    user: dict = Depends(get_current_user),
):
    """Generate a PDF or DOCX file from Markdown content.

    The Banamex logo is placed in the top-left header of every page.
    Returns raw file bytes with the correct Content-Type and
    Content-Disposition headers so the browser triggers a download.
    """
    from services.exporters import get_exporter, SUPPORTED_FORMATS

    fmt = req.format.lower().strip()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{fmt}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )

    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    title = (req.title or "satriani-export").strip() or "satriani-export"
    # Sanitise title for use in Content-Disposition filename
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)
    safe_title = safe_title.strip().replace(" ", "_")[:80] or "satriani-export"

    try:
        exporter = get_exporter(fmt)
        file_bytes = exporter.export(req.content, title)
    except Exception as exc:
        logger.error("Export error (format=%s): %s", fmt, exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}")

    filename = f"{safe_title}{exporter.extension}"
    return Response(
        content=file_bytes,
        media_type=exporter.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(file_bytes)),
        },
    )
