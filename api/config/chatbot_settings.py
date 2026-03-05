'''
Chatbot Configuration — Satriani AI Platform

Models must be downloaded once and stored in MODELS_BASE_DIR.
After that, the application runs fully offline.

Chat models:
  - Gemma-3-1B-IT            (~3.8GB)
  - Meta-Llama-3.1-8B-Instruct (~16GB)

Embedding models:
  - gte-large-en-v1.5   (8192 ctx)
  - gte-multilingual-base (8192 ctx)

Compatible with Python 3.9.20.
'''

import os

# ─── Models Base Directory ────────────────────────────────────────────────────
MODELS_BASE_DIR = os.environ.get(
    "MODELS_BASE_DIR",
    "/home/ivan/ProjectPrometheus/models"
)

# ─── Chat Models ──────────────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "gemma3-1b": {
        "hf_id": "google/gemma-3-1b-it",
        "local_dir": os.path.join(MODELS_BASE_DIR, "gemma-3-1b-it"),
        "display_name": "Gemma 3 1B",
        "size": "~3.8 GB",
        "context_length": 8192,
        "description": "Compact and fast. Good quality for its size.",
    },
    "llama3.1-8b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "local_dir": os.path.join(MODELS_BASE_DIR, "Meta-Llama-3.1-8B-Instruct"),
        "display_name": "Llama 3.1 8B Instruct",
        "size": "~16 GB",
        "context_length": 32000,
        "description": "High quality. 32K context window.",
    },
}

# ─── Embedding Models ─────────────────────────────────────────────────────────
AVAILABLE_EMBEDDING_MODELS = {
    "gte-large-en": {
        "hf_id": "Alibaba-NLP/gte-large-en-v1.5",
        "local_dir": os.path.join(MODELS_BASE_DIR, "gte-large-en-v1.5"),
        "display_name": "GTE Large EN v1.5",
        "context_length": 8192,
        "description": "High quality English embeddings.",
    },
    "gte-multilingual": {
        "hf_id": "Alibaba-NLP/gte-multilingual-base",
        "local_dir": os.path.join(MODELS_BASE_DIR, "gte-multilingual-base"),
        "display_name": "GTE Multilingual Base",
        "context_length": 8192,
        "description": "Multilingual embeddings (EN, ES, etc.).",
    },
}

DEFAULT_MODEL_KEY = os.environ.get("DEFAULT_MODEL", "gemma3-1b")
DEFAULT_EMBEDDING_KEY = os.environ.get("DEFAULT_EMBEDDING", "gte-large-en")

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ─── Generation Parameters ───────────────────────────────────────────────────
CHATBOT_MAX_NEW_TOKENS = 1024
CHATBOT_TEMPERATURE = 0.7
CHATBOT_TOP_P = 0.9
CHATBOT_TOP_K = 50
CHATBOT_REPETITION_PENALTY = 1.3
CHATBOT_DO_SAMPLE = True

# ─── Data Configuration ──────────────────────────────────────────────────────
CHATBOT_CSV_PATH = "static/data/cubo_datos_v2.csv"
CHATBOT_MAX_CONTEXT_ROWS = 50
CHATBOT_TOP_N_CATEGORIES = 10

# ─── RAG Data Directory ──────────────────────────────────────────────────────
RAG_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)
RAG_UPLOAD_MAX_SIZE_MB = 5

# ─── ChromaDB Persist Directory ──────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chroma_db"
)

# ─── Auth ─────────────────────────────────────────────────────────────────────
AUTH_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "db", "users.json"
)
# SQLite database for users, sessions, documents
SQLITE_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "db", "satriani.db"
)
ADMIN_DEFAULT_USERNAME = "admin"
ADMIN_DEFAULT_PASSWORD = "satriani2025"
SESSION_SECRET = os.environ.get("SESSION_SECRET", "satriani-secret-key-change-me")

# ─── System Prompt ────────────────────────────────────────────────────────────
CHATBOT_SYSTEM_PROMPT = (
    "You are Satriani, an expert AI assistant specialized in document analysis, "
    "data interpretation, and knowledge retrieval.\n\n"
    "FUNDAMENTAL RULE: ALWAYS respond in English regardless of the "
    "language of the question.\n\n"
    "INSTRUCTIONS:\n"
    "1. ALWAYS respond in English\n"
    "2. Base your answer on the context provided when available\n"
    "3. Be precise with numbers, use commas for thousands\n"
    "4. If you cannot answer from the context, say so clearly\n"
    "5. Provide insights and explain patterns when relevant\n"
    "6. Use structured formatting with bullet points\n"
)
