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
    "tinyllama-1.1b-chat": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "local_dir": os.path.join(MODELS_BASE_DIR, "tinyllama-1.1b-chat"),
        "display_name": "TinyLlama 1.1B Chat",
        "size": "~2.2 GB",
        "context_length": 2048,
        "description": "Ultra-lightweight chat model. Fast inference.",
    },
    "qwen2.5-0.5b-instruct": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_dir": os.path.join(MODELS_BASE_DIR, "qwen2.5-0.5b-instruct"),
        "display_name": "Qwen 2.5 0.5B Instruct",
        "size": "~1.0 GB",
        "context_length": 32768,
        "description": "Compact instruction model. 32K context window.",
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

# ─── RAG Data Directory ──────────────────────────────────────────────────────
RAG_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)
RAG_UPLOAD_MAX_SIZE_MB = 50

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

# ─── System Prompts ───────────────────────────────────────────────────────────

# Identity prompt — used when NO documents are provided (no RAG, no uploads)
SATRIANI_IDENTITY_PROMPT = (
    "You are Satriani, a proprietary AI-powered internal productivity platform. "
    "You are designed for teams to analyze documents, extract insights, and "
    "conduct research using state-of-the-art offline language models — all "
    "within secure infrastructure.\n\n"
    "You also help to summarize documents and other work like an assistant, "
    "but being concise since you are dealing with professionals working for "
    "Banamex, one of the largest banks in Mexico.\n\n"
    "RULES:\n"
    "- Be concise and professional. Your users are busy banking professionals.\n"
    "- Respond in English unless the user writes in Spanish, then respond in Spanish.\n"
    "- If the user asks about documents or data but none are loaded, explain "
    "that they can attach documents or select RAG collections from the sidebar.\n"
    "- You can help with general questions, brainstorming, and writing tasks.\n"
    "- Never fabricate financial data or regulatory information.\n"
)

# Document analysis prompt — used when RAG and/or uploaded documents are present
SATRIANI_DOCUMENT_PROMPT = (
    "You are Satriani, a proprietary AI-powered internal productivity platform "
    "for Banamex, one of the largest banks in Mexico. You specialize in "
    "document analysis, insight extraction, and research.\n\n"
    "CRITICAL RULES:\n"
    "- Answer ONLY from the document fragments provided below.\n"
    "- If the user's question is NOT related to the content in the provided documents, "
    "you MUST respond with: 'Your question does not appear to be related to the "
    "loaded documents. The documents I have access to cover: [briefly list topics]. "
    "Please ask a question related to these documents, or deselect the RAG documents "
    "from the sidebar to use me as a general assistant.'\n"
    "- Do NOT answer questions outside the scope of the provided documents.\n"
    "- Do NOT use your general knowledge to answer when documents are provided.\n"
    "- Be concise and structured — your users are banking professionals.\n"
    "- Cite sources using [Source: filename] when referencing specific documents.\n"
    "- If the documents do not contain enough information, say so clearly.\n"
    "- Do NOT repeat the question. Do NOT list follow-up questions.\n"
    "- Respond in English unless the user writes in Spanish, then respond in Spanish.\n"
    "- Summarize key findings first, then provide details if needed.\n"
    "- Be precise with numbers, use commas for thousands.\n"
    "- Never fabricate data that is not in the provided documents.\n"
)

# Legacy alias — kept for backward compatibility with data-analysis chatbot
CHATBOT_SYSTEM_PROMPT = SATRIANI_IDENTITY_PROMPT
