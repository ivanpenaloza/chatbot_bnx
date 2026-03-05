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

---

## Database Schema

Satriani uses two SQLite databases: one application-level database for users, sessions, and document metadata, and one ChromaDB-managed database for vector embeddings and RAG retrieval.

### 1. Application Database — `api/db/satriani.db`

Managed by `api/db.py`. Uses WAL journal mode and foreign keys.

#### `users`
| Column | Type | Constraints | Description |
|---|---|---|---|
| `username` | TEXT | PRIMARY KEY | Unique login identifier |
| `password_hash` | TEXT | NOT NULL | SHA-256 hash (password + session secret) |
| `role` | TEXT | NOT NULL, DEFAULT `'user'` | `admin` or `user` |
| `full_name` | TEXT | NOT NULL, DEFAULT `''` | Display name |
| `created_at` | TEXT | NOT NULL, DEFAULT `datetime('now')` | ISO-8601 creation timestamp |

#### `sessions`
| Column | Type | Constraints | Description |
|---|---|---|---|
| `token` | TEXT | PRIMARY KEY | 64-char hex token (`secrets.token_hex(32)`) |
| `username` | TEXT | NOT NULL, FK → `users.username` ON DELETE CASCADE | Owning user |
| `role` | TEXT | NOT NULL | Cached role at login time |
| `full_name` | TEXT | NOT NULL, DEFAULT `''` | Cached display name |
| `created_at` | TEXT | NOT NULL, DEFAULT `datetime('now')` | Session creation timestamp |

#### `documents`
| Column | Type | Constraints | Description |
|---|---|---|---|
| `filename` | TEXT | PRIMARY KEY | Original filename (e.g. `GuidelineForAIandMLModels.docx`) |
| `size_kb` | REAL | NOT NULL, DEFAULT `0` | File size in kilobytes |
| `uploaded_at` | TEXT | NOT NULL, DEFAULT `datetime('now')` | Upload timestamp |
| `uploaded_by` | TEXT | NOT NULL, DEFAULT `'system'` | Username of uploader |

**Example rows (documents):**
```
filename                                  size_kb   uploaded_at           uploaded_by
ConsumerValuationModelsMDDT.docx          447.0     2026-03-04 15:26:41   system
GuidelineForAIandMLModels.docx            245.0     2026-03-04 15:26:41   system
GuidelineSponsorsCVandAMLModels.docx      5685.1    2026-03-04 15:26:41   system
hopkinsmedicine_CFS-low-histamine-diet.pdf 141.0    2026-03-04 20:43:02   admin
```

### 2. ChromaDB Vector Store — `api/chroma_db/chroma.sqlite3`

Managed internally by ChromaDB's `PersistentClient`. The application interacts via the ChromaDB Python API — never by raw SQL. The schema below is documented for reference and debugging only.

#### Core tables

| Table | Purpose |
|---|---|
| `tenants` | Multi-tenant isolation (single `default_tenant`) |
| `databases` | Logical databases (single `default_database`) |
| `collections` | One collection per ingested document × embedding model |
| `collection_metadata` | Key-value metadata per collection (`embedding_model`, `source_file`) |
| `segments` | Storage segments: VECTOR (HNSW) + METADATA (SQLite) per collection |
| `segment_metadata` | Key-value metadata per segment |

#### `collections` (key table)
| Column | Type | Description |
|---|---|---|
| `id` | TEXT PK | UUID |
| `name` | TEXT | Collection name, e.g. `ConsumerValuationModelsMDDT` or `hopkinsmedicine_CFS-low-histamine-diet__gte-large-en` |
| `dimension` | INTEGER | Embedding dimension (1024 for GTE-Large, 768 for GTE-Multilingual) |
| `database_id` | TEXT | FK to `databases.id` |
| `config_json_str` | TEXT | ChromaDB config JSON |

**Current collections (5):**
| Name | Dimension | Embedding Model |
|---|---|---|
| `ConsumerValuationModelsMDDT` | 1024 | gte-large-en |
| `GuidelineForAIandMLModels` | 768 | gte-multilingual |
| `GuidelineSponsorsCVandAMLModels` | 768 | gte-multilingual |
| `hopkinsmedicine_CFS-low-histamine-diet__gte-large-en` | 1024 | gte-large-en |
| `hopkinsmedicine_CFS-low-histamine-diet__gte-multilingual` | 768 | gte-multilingual |

#### `collection_metadata`
Key-value pairs per collection. Current keys: `embedding_model`, `source_file`.

#### Embedding tables

| Table | Purpose |
|---|---|
| `embeddings` | Maps `embedding_id` → `segment_id` with sequence and timestamp |
| `embedding_metadata` | Per-chunk metadata: `chunk_index` (int), `source` (string), `chroma:document` (full chunk text) |
| `embedding_fulltext_search` | FTS5 virtual table for full-text search over chunk text |
| `embeddings_queue` | Write-ahead queue with raw vectors (FLOAT32 blobs), operation log |

**Embedding ID convention:** `{CollectionName}_{chunkIndex}_{hash8}` (e.g. `ConsumerValuationModelsMDDT_0_0eca04e3`)

**Current stats:** 887 total embeddings across 5 collections.

#### Segment types
Each collection has two segments:
- `urn:chroma:segment/vector/hnsw-local-persisted` — HNSW index stored in `api/chroma_db/{segment_uuid}/` (binary files: `data_level0.bin`, `header.bin`, `length.bin`, `link_lists.bin`)
- `urn:chroma:segment/metadata/sqlite` — Metadata stored in the same `chroma.sqlite3`

#### Internal / housekeeping tables
| Table | Purpose |
|---|---|
| `migrations` | Schema version tracking (18 migrations applied) |
| `max_seq_id` | Per-segment sequence watermark |
| `embeddings_queue_config` | Queue settings (`automatically_purge: true`) |
| `maintenance_log` | Maintenance operations log (currently empty) |
| `acquire_write` | Write lock coordination |

---

## Running

```bash
cd api
PORT=8000 python main.py
```

Default admin credentials: `admin` / `satriani2025`
