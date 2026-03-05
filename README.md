# Satriani — AI Document Intelligence Platform

An AI-powered internal productivity platform for document analysis and knowledge retrieval. Runs fully offline using locally-stored LLM and embedding models — no data leaves the infrastructure.

## Features

- Landing page with user/admin login
- Session-based authentication with admin/user roles
- Admin console: user management, document upload (.docx/.pdf), embedding model selection, ingestion lifecycle
- Unified chat interface: RAG-grounded answers with source citations, temporary context upload (max 5MB), free chat mode
- RAG pipeline: ChromaDB (SQLite-backed), 500-char overlapping chunks, top-3 retrieval
- OpenEvidence-style expandable source cards

## Tech Stack

- Python 3.9.20, FastAPI + Uvicorn
- ChromaDB (PersistentClient, SQLite backend)
- sentence-transformers for embeddings
- transformers + torch for LLM inference
- python-docx + PyPDF2 for document extraction

## Models

| Type | Model | Size | Context |
|------|-------|------|---------|
| Chat | google/gemma-3-1b-it | ~3.8 GB | — |
| Chat | meta-llama/Meta-Llama-3.1-8B-Instruct | ~16 GB | 32K |
| Embedding | Alibaba-NLP/gte-large-en-v1.5 | — | 8192 |
| Embedding | Alibaba-NLP/gte-multilingual-base | — | 8192 |

## Quick Start

```bash
# Install dependencies
python39 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Run
cd api
PORT=8000 python main.py
```

Default admin credentials: `admin` / `satriani2025`

---

## Database Schema

Satriani uses two SQLite databases: an application database for users, sessions, and document metadata, and a ChromaDB-managed database for vector embeddings.

### 1. Application Database — `api/db/satriani.db`

Managed by `api/db.py`. Uses WAL journal mode and foreign keys enabled.

#### `users`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `username` | TEXT | PRIMARY KEY | Unique login identifier |
| `password_hash` | TEXT | NOT NULL | SHA-256 hash (password + session secret) |
| `role` | TEXT | NOT NULL, DEFAULT `'user'` | `admin` or `user` |
| `full_name` | TEXT | NOT NULL, DEFAULT `''` | Display name |
| `created_at` | TEXT | NOT NULL, DEFAULT `datetime('now')` | ISO-8601 creation timestamp |

#### `sessions`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `token` | TEXT | PRIMARY KEY | 64-char hex session token |
| `username` | TEXT | NOT NULL, FK → `users.username` ON DELETE CASCADE | Owning user |
| `role` | TEXT | NOT NULL | Cached role at login time |
| `full_name` | TEXT | NOT NULL, DEFAULT `''` | Cached display name |
| `created_at` | TEXT | NOT NULL, DEFAULT `datetime('now')` | Session creation timestamp |

#### `documents`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `filename` | TEXT | PRIMARY KEY | Original filename |
| `size_kb` | REAL | NOT NULL, DEFAULT `0` | File size in kilobytes |
| `uploaded_at` | TEXT | NOT NULL, DEFAULT `datetime('now')` | Upload timestamp |
| `uploaded_by` | TEXT | NOT NULL, DEFAULT `'system'` | Username of uploader |

Example data:

```
filename                                   size_kb   uploaded_by
ConsumerValuationModelsMDDT.docx           447.0     system
GuidelineForAIandMLModels.docx             245.0     system
GuidelineSponsorsCVandAMLModels.docx       5685.1    system
hopkinsmedicine_CFS-low-histamine-diet.pdf 141.0     admin
```

### 2. ChromaDB Vector Store — `api/chroma_db/chroma.sqlite3`

Managed internally by ChromaDB's `PersistentClient`. The application interacts via the ChromaDB Python API — never by raw SQL. Schema documented here for reference and debugging.

#### Core tables

| Table | Purpose |
|-------|---------|
| `tenants` | Multi-tenant isolation (single `default_tenant`) |
| `databases` | Logical databases (single `default_database`) |
| `collections` | One collection per ingested document × embedding model |
| `collection_metadata` | Key-value metadata per collection (`embedding_model`, `source_file`) |
| `segments` | Storage segments: VECTOR (HNSW) + METADATA (SQLite) per collection |

#### `collections`

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `name` | TEXT | Collection name (document name or `{doc}__{model}`) |
| `dimension` | INTEGER | Embedding vector dimension |
| `database_id` | TEXT | FK to `databases.id` |
| `config_json_str` | TEXT | ChromaDB config JSON |

Current collections:

| Name | Dimension | Embedding Model |
|------|-----------|-----------------|
| `ConsumerValuationModelsMDDT` | 1024 | gte-large-en |
| `GuidelineForAIandMLModels` | 768 | gte-multilingual |
| `GuidelineSponsorsCVandAMLModels` | 768 | gte-multilingual |
| `hopkinsmedicine_CFS-low-histamine-diet__gte-large-en` | 1024 | gte-large-en |
| `hopkinsmedicine_CFS-low-histamine-diet__gte-multilingual` | 768 | gte-multilingual |

#### Embedding tables

| Table | Purpose |
|-------|---------|
| `embeddings` | Maps `embedding_id` → `segment_id` with sequence and timestamp |
| `embedding_metadata` | Per-chunk metadata: `chunk_index`, `source`, `chroma:document` (chunk text) |
| `embedding_fulltext_search` | FTS5 virtual table for full-text search over chunk text |
| `embeddings_queue` | Write-ahead queue with raw FLOAT32 vectors |

Embedding ID convention: `{CollectionName}_{chunkIndex}_{hash8}`

Current stats: 887 total embeddings across 5 collections.

#### Segment types

Each collection has two segments:
- `urn:chroma:segment/vector/hnsw-local-persisted` — HNSW index in `api/chroma_db/{segment_uuid}/` (binary files)
- `urn:chroma:segment/metadata/sqlite` — Metadata in the same `chroma.sqlite3`

#### Internal tables

| Table | Purpose |
|-------|---------|
| `migrations` | Schema version tracking (18 migrations) |
| `max_seq_id` | Per-segment sequence watermark |
| `embeddings_queue_config` | Queue settings (`automatically_purge: true`) |
| `maintenance_log` | Maintenance operations log |
| `acquire_write` | Write lock coordination |

---

## Project Structure

```
chatbot_bnx/
├── api/
│   ├── main.py                    # FastAPI entry point, lifespan, router wiring
│   ├── db.py                      # SQLite database layer (users, sessions, documents)
│   ├── config/
│   │   ├── default_settings.py    # Infrastructure config (ports, paths)
│   │   ├── chatbot_settings.py    # Model registry, generation params, RAG config, auth
│   │   └── jinja_functions.py     # Custom Jinja2 template functions
│   ├── routers/
│   │   ├── auth_routes.py         # Authentication (login, register, sessions)
│   │   ├── admin_routes.py        # Admin panel (upload docs, embeddings, ingestion)
│   │   ├── llm_routes.py          # LLM model management (load, switch, generate)
│   │   └── rag_routes.py          # RAG pipeline (ingest, query, unified chat)
│   ├── db/                        # satriani.db (SQLite application database)
│   ├── chroma_db/                 # ChromaDB persistent storage
│   ├── data/                      # Uploaded RAG documents (.docx, .pdf)
│   ├── templates/                 # Jinja2 HTML templates
│   └── static/                    # Static assets (images, legacy data)
├── docs/                          # Context files, prompts, schema docs
├── requirements.txt
└── .github/
    └── copilot-instructions.md
```

---

See also: [FastAPI docs](https://fastapi.tiangolo.com/)
