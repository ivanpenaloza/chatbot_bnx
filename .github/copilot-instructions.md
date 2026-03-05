# Satriani — AI Document Intelligence Platform

## Overview

Satriani is a proprietary, AI-powered internal productivity platform for document analysis and knowledge retrieval. It runs fully offline using locally-stored LLM and embedding models — no data leaves the infrastructure.

## Architecture

```
chatbot_bnx/
├── api/
│   ├── main.py                    # FastAPI app entry point, lifespan, router wiring
│   ├── config/
│   │   ├── __init__.py            # Exports all settings
│   │   ├── default_settings.py    # Infrastructure config (ports, paths, Hadoop, Spark)
│   │   ├── chatbot_settings.py    # Model registry, generation params, RAG config, auth
│   │   └── jinja_functions.py     # Custom Jinja2 template functions
│   ├── routers/
│   │   ├── auth_routes.py         # Authentication (login, register, sessions, user mgmt)
│   │   ├── admin_routes.py        # Admin panel (upload docs, embedding models, ingestion)
│   │   ├── llm_routes.py          # LLM model management (load, switch, generate, health)
│   │   └── rag_routes.py          # RAG pipeline (ingest, query, unified chat, upload context)
│   ├── templates/
│   │   ├── landing.html           # Landing page with user/admin login
│   │   ├── admin.html             # Admin console (users, documents, embeddings)
│   │   ├── rag_chat.html          # User chat interface (unified RAG + free)
│   │   └── chatbot.html           # Legacy chatbot (data-focused)
│   └── static/
│       ├── data/                  # RAG documents (.docx, .pdf), ChromaDB, CSV data
│       └── images/                # Logos and icons
├── docs/                          # Context files, prompts, schema docs
├── requirements.txt               # Python dependencies
└── .github/
    └── copilot-instructions.md    # This file
```

## Key Features

1. **Landing Page** — Compelling intro with user/admin login options
2. **Authentication** — JSON-file-based user store, session tokens, admin/user roles
3. **Admin Console** — User management, document upload (.docx/.pdf), ingestion with embedding model selection, document lifecycle
4. **Unified Chat** — Single chat interface where users can:
   - Select RAG documents for grounded answers with source citations
   - Upload temporary context documents (max 5MB)
   - Chat freely without any context (general knowledge)
5. **RAG Pipeline** — ChromaDB (SQLite-backed), text extraction (docx + pdf), 500-char overlapping chunks, top-3 retrieval
6. **OpenEvidence-style Sources** — Expandable source cards showing document fragments used to answer

## Models

### Chat Models (one loaded at a time)
- **Gemma 3 1B** — google/gemma-3-1b-it (~3.8 GB)
- **Llama 3.1 8B Instruct** — meta-llama/Meta-Llama-3.1-8B-Instruct (~16 GB, 32K ctx)

### Embedding Models
- **GTE Large EN v1.5** — Alibaba-NLP/gte-large-en-v1.5 (8192 ctx)
- **GTE Multilingual Base** — Alibaba-NLP/gte-multilingual-base (8192 ctx)

## Tech Stack

- **Python 3.9.20** (strict compatibility requirement)
- **FastAPI** + Uvicorn
- **ChromaDB** (PersistentClient, SQLite backend)
- **sentence-transformers** for embeddings
- **transformers + torch** for LLM inference
- **python-docx** + **PyPDF2** for document extraction

## Running

```bash
cd api
PORT=8000 python main.py
```

Default admin credentials: `admin` / `satriani2025`
