"""
RAG Router — Satriani Document Chat

Provides REST API endpoints for a RAG-powered chatbot that:
  1. Ingests .docx files from static/data → extracts text
  2. Chunks and embeds text using a local sentence-transformers model
  3. Stores embeddings in ChromaDB (SQLite-backed persistent storage)
  4. Retrieves top-3 relevant chunks per query
  5. Feeds them as context to the same offline LLM models used by llm_routes

Compatible with Python 3.9.20.
"""

import os
import sys
import re
import logging
import hashlib
import traceback
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import MODELS_BASE_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["rag"])
templates = Jinja2Templates(directory="templates")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "data")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "data", "chroma_db")
EMBEDDING_MODEL_PATH = os.path.join(MODELS_BASE_DIR, "all-MiniLM-L6-v2")

# ─── Globals (lazy-loaded) ───────────────────────────────────────────────────
_chroma_client = None
_embedding_fn = None


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class RAGChatMessage(BaseModel):
    message: str
    collections: Optional[List[str]] = None  # filter by doc collections


class RAGIngestRequest(BaseModel):
    filename: Optional[str] = None  # ingest specific file, or all if None


# ─── Embedding Function ──────────────────────────────────────────────────────

class LocalHFEmbeddingFunction:
    """ChromaDB-compatible embedding function using local sentence-transformers.

    Implements the interface expected by chromadb >= 1.5:
    __call__, name, embed_query, and related methods.
    """

    def __init__(self, model_path: str):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self._model = SentenceTransformer(model_path)
        self._name = "local-hf-" + os.path.basename(model_path)
        self._np = np
        logger.info("Embedding model loaded from: %s", model_path)

    def name(self) -> str:
        return self._name

    def __call__(self, input: List[str]) -> List:
        embeddings = self._model.encode(input, show_progress_bar=False)
        return [self._np.array(e, dtype=self._np.float32) for e in embeddings]

    def embed_query(self, input: List[str]) -> List:
        return self.__call__(input)

    def embed_with_retries(self, input, **kwargs):
        return self.__call__(input)

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self):
        return ["cosine", "l2", "ip"]

    def get_config(self):
        return {"model_path": self._name}

    @staticmethod
    def validate_config(config):
        pass

    def validate_config_update(self, old_config, new_config):
        pass

    @classmethod
    def build_from_config(cls, config):
        raise NotImplementedError

    def is_legacy(self) -> bool:
        return False


def get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = LocalHFEmbeddingFunction(EMBEDDING_MODEL_PATH)
    return _embedding_fn


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        from chromadb.config import Settings
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB persistent client initialized at: %s", CHROMA_PERSIST_DIR)
    return _chroma_client


# ─── DOCX Text Extraction ────────────────────────────────────────────────────

def extract_text_from_docx(filepath: str) -> str:
    """Extract all text from a .docx file using python-docx."""
    from docx import Document
    doc = Document(filepath)
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    # Also extract from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)
    return "\n".join(paragraphs)


# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def sanitize_collection_name(name: str) -> str:
    """Make a valid ChromaDB collection name from a filename."""
    # Remove extension, replace non-alphanumeric with underscore
    base = os.path.splitext(name)[0]
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', base)
    # ChromaDB requires 3-63 chars, must start/end with alphanumeric
    sanitized = sanitized.strip('_-')
    if len(sanitized) < 3:
        sanitized = sanitized + "_doc"
    if len(sanitized) > 63:
        sanitized = sanitized[:63].rstrip('_-')
    return sanitized


# ─── Ingestion Logic ─────────────────────────────────────────────────────────

def ingest_docx_file(filepath: str) -> Dict[str, Any]:
    """Ingest a single .docx file into ChromaDB."""
    filename = os.path.basename(filepath)
    collection_name = sanitize_collection_name(filename)

    logger.info("Ingesting: %s → collection: %s", filename, collection_name)

    # Extract text
    text = extract_text_from_docx(filepath)
    if not text.strip():
        return {"filename": filename, "status": "empty", "chunks": 0}

    # Chunk
    chunks = chunk_text(text, chunk_size=500, overlap=100)

    # Get or create collection
    client = get_chroma_client()
    ef = get_embedding_fn()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"source_file": filename},
    )

    # Check if already ingested (by count)
    if collection.count() > 0:
        logger.info("Collection '%s' already has %d chunks, skipping.", collection_name, collection.count())
        return {
            "filename": filename,
            "collection": collection_name,
            "status": "already_ingested",
            "chunks": collection.count(),
        }

    # Generate IDs and add
    ids = []
    for i, chunk in enumerate(chunks):
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
        ids.append(f"{collection_name}_{i}_{chunk_hash}")

    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    # Add in batches of 100
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info("Ingested %d chunks for '%s'", len(chunks), filename)
    return {
        "filename": filename,
        "collection": collection_name,
        "status": "ingested",
        "chunks": len(chunks),
        "text_length": len(text),
    }


def get_all_docx_files() -> List[str]:
    """Return list of .docx files in DATA_DIR."""
    files = []
    if os.path.isdir(DATA_DIR):
        for f in sorted(os.listdir(DATA_DIR)):
            if f.lower().endswith('.docx'):
                files.append(os.path.join(DATA_DIR, f))
    return files


# ─── Query Logic ──────────────────────────────────────────────────────────────

def query_collections(
    question: str,
    collection_names: Optional[List[str]] = None,
    n_results: int = 3,
) -> List[Dict[str, Any]]:
    """Query ChromaDB collections and return top-N relevant chunks."""
    client = get_chroma_client()
    ef = get_embedding_fn()

    # Determine which collections to search
    all_collections = client.list_collections()
    if collection_names:
        target_names = collection_names
    else:
        target_names = [c.name for c in all_collections]

    if not target_names:
        return []

    results = []
    for cname in target_names:
        try:
            collection = client.get_collection(name=cname, embedding_function=ef)
            if collection.count() == 0:
                continue
            query_result = collection.query(
                query_texts=[question],
                n_results=min(n_results, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            docs = query_result.get("documents", [[]])[0]
            metas = query_result.get("metadatas", [[]])[0]
            dists = query_result.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                results.append({
                    "collection": cname,
                    "document": doc,
                    "source": meta.get("source", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": round(dist, 4),
                })
        except Exception as e:
            logger.warning("Error querying collection '%s': %s", cname, e)

    # Sort by distance (lower = more relevant) and take top N
    results.sort(key=lambda x: x["distance"])
    return results[:n_results]


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/rag/chat", response_class=HTMLResponse)
async def rag_chat_page(request: Request):
    """Serve the Satriani RAG chat page."""
    return templates.TemplateResponse("rag_chat.html", {"request": request})


@router.post("/rag/ingest", response_class=JSONResponse)
async def rag_ingest(req: RAGIngestRequest):
    """Ingest .docx files into ChromaDB collections."""
    try:
        files = get_all_docx_files()
        if not files:
            return JSONResponse(content={"status": "no_files", "message": "No .docx files found in data directory."})

        if req.filename:
            files = [f for f in files if os.path.basename(f) == req.filename]
            if not files:
                raise HTTPException(status_code=404, detail=f"File not found: {req.filename}")

        results = []
        for fpath in files:
            result = ingest_docx_file(fpath)
            results.append(result)

        return JSONResponse(content={"status": "ok", "results": results})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ingestion error: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/collections", response_class=JSONResponse)
async def rag_list_collections():
    """List all ingested document collections."""
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        result = []
        for c in collections:
            col = client.get_collection(name=c.name)
            meta = col.metadata or {}
            result.append({
                "name": c.name,
                "source_file": meta.get("source_file", "Unknown"),
                "chunks": col.count(),
            })
        return JSONResponse(content={"collections": result})
    except Exception as e:
        logger.error("Error listing collections: %s", e)
        return JSONResponse(content={"collections": []})


@router.get("/rag/documents", response_class=JSONResponse)
async def rag_list_documents():
    """List available .docx files and their ingestion status."""
    files = get_all_docx_files()
    client = get_chroma_client()
    existing = {c.name for c in client.list_collections()}

    docs = []
    for fpath in files:
        fname = os.path.basename(fpath)
        cname = sanitize_collection_name(fname)
        ingested = cname in existing
        chunk_count = 0
        if ingested:
            try:
                col = client.get_collection(name=cname)
                chunk_count = col.count()
            except Exception:
                pass
        docs.append({
            "filename": fname,
            "collection": cname,
            "ingested": ingested,
            "chunks": chunk_count,
            "size_kb": round(os.path.getsize(fpath) / 1024, 1),
        })
    return JSONResponse(content={"documents": docs})


@router.post("/rag/ask", response_class=JSONResponse)
async def rag_ask(chat: RAGChatMessage):
    """Answer a question using RAG: retrieve top-3 chunks then generate with LLM."""
    if not chat.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Retrieve relevant chunks
    chunks = query_collections(
        question=chat.message,
        collection_names=chat.collections,
        n_results=3,
    )

    if not chunks:
        return JSONResponse(content={
            "response": "No relevant documents found. Please ingest documents first.",
            "sources": [],
            "inference_time": 0,
        })

    # Build context from retrieved chunks
    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"--- Fragment {i} (from: {chunk['source']}, relevance: {1 - chunk['distance']:.2%}) ---\n"
            f"{chunk['document']}"
        )
        sources.append({
            "source": chunk["source"],
            "collection": chunk["collection"],
            "chunk_index": chunk["chunk_index"],
            "relevance": round(1 - chunk["distance"], 4),
            "text": chunk["document"],
        })

    rag_context = "\n\n".join(context_parts)

    # Build system prompt for RAG
    system_prompt = (
        "You are Satriani, an expert document analysis assistant.\n\n"
        "FUNDAMENTAL RULE: ALWAYS respond in English regardless of "
        "the language of the question.\n\n"
        "You are provided with relevant fragments from internal documents. "
        "Use ONLY the information from these fragments to answer.\n"
        "If the information is insufficient, state so clearly.\n"
        "Cite the source document when possible.\n"
        "Respond in a structured and professional manner.\n"
    )

    # Use the same LLM model manager from llm_routes
    from .llm_routes import model_manager

    answer, inference_time = model_manager.generate_response(
        user_message=chat.message,
        data_context=system_prompt + "\n\nRELEVANT DOCUMENTS:\n" + rag_context,
    )

    return JSONResponse(content={
        "response": answer,
        "sources": sources,
        "inference_time": inference_time,
        "chunks_used": len(chunks),
    })


@router.post("/rag/free-ask", response_class=JSONResponse)
async def rag_free_ask(chat: RAGChatMessage):
    """Free mode: answer using only the pre-trained LLM, no RAG context."""
    if not chat.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    from .llm_routes import model_manager

    system_prompt = (
        "You are Satriani, a knowledgeable AI assistant.\n\n"
        "FUNDAMENTAL RULE: ALWAYS respond in English regardless of "
        "the language of the question.\n\n"
        "Answer the user's question using your general knowledge.\n"
        "Be precise, structured, and professional.\n"
        "If you are unsure about something, say so clearly.\n"
    )

    answer, inference_time = model_manager.generate_response(
        user_message=chat.message,
        data_context=system_prompt,
    )

    return JSONResponse(content={
        "response": answer,
        "sources": [],
        "inference_time": inference_time,
        "chunks_used": 0,
    })
