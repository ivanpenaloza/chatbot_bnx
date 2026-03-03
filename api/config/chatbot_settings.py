'''
Chatbot Configuration - Multi-Model Offline LLM Support

Configuration settings for the AI-powered chatbot that loads
locally-stored LLM models to answer statistical questions
about the cubo_datos_v2 dataset.

Models must be downloaded once using the provided download script
and stored in the MODELS_BASE_DIR. After that, the application
runs fully offline with no internet dependency.

Supported models:
  - Qwen2.5-0.5B-Instruct  (~1GB, fastest)
  - TinyLlama-1.1B-Chat     (~2.2GB, fast)
  - Gemma-3-1B-IT            (~3.8GB, best quality)

Authors: Ivan Dario Penaloza Rojas <ip70574@citi.com>
Manager: Ivan Dario Penaloza Rojas <ip70574@citi.com>
'''

import os

# ─── Models Base Directory ────────────────────────────────────────────────────
MODELS_BASE_DIR = os.environ.get(
    "MODELS_BASE_DIR",
    "/home/ivan/ProjectPrometheus/models"
)

# ─── Available Models Registry ───────────────────────────────────────────────
# Each model entry: key -> {hf_id, local_dir, display_name, size, description}
AVAILABLE_MODELS = {
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_dir": os.path.join(MODELS_BASE_DIR, "qwen2.5-0.5b-instruct"),
        "display_name": "Qwen 2.5 0.5B",
        "size": "~1 GB",
        "description": "Fastest. Great for quick answers.",
    },
    "tinyllama-1.1b": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "local_dir": os.path.join(MODELS_BASE_DIR, "tinyllama-1.1b-chat"),
        "display_name": "TinyLlama 1.1B",
        "size": "~2.2 GB",
        "description": "Fast and balanced performance.",
    },
    "gemma3-1b": {
        "hf_id": "google/gemma-3-1b-it",
        "local_dir": os.path.join(MODELS_BASE_DIR, "gemma-3-1b-it"),
        "display_name": "Gemma 3 1B",
        "size": "~3.8 GB",
        "description": "Best quality. Slower inference.",
    },
}

# Default model to load at startup
DEFAULT_MODEL_KEY = os.environ.get("DEFAULT_MODEL", "qwen2.5-0.5b")

# HuggingFace token (only needed during model download, not at runtime)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ─── Generation Parameters ───────────────────────────────────────────────────
CHATBOT_MAX_NEW_TOKENS = 1024
CHATBOT_TEMPERATURE = 0.7
CHATBOT_TOP_P = 0.9
CHATBOT_TOP_K = 50
CHATBOT_REPETITION_PENALTY = 1.15
CHATBOT_DO_SAMPLE = True

# ─── Data Configuration ──────────────────────────────────────────────────────
CHATBOT_CSV_PATH = "static/data/cubo_datos_v2.csv"
CHATBOT_MAX_CONTEXT_ROWS = 50
CHATBOT_TOP_N_CATEGORIES = 10

# ─── System Prompt ────────────────────────────────────────────────────────────
CHATBOT_SYSTEM_PROMPT = (
    "You are Satriani, an expert AI assistant specialized in data analysis, "
    "financial products, and credit campaign analytics.\n\n"
    "FUNDAMENTAL RULE: ALWAYS respond in English regardless of the "
    "language of the question.\n\n"
    "You have access to a dataset called 'cubo_datos_v2' containing "
    "campaign assignment and financial product data with these columns:\n"
    "- etiqueta_grupo: Campaign groups and eligibility\n"
    "- producto: Financial products (AAC, CLI, CPC, DC, TCC, CNC, CPT, CNT)\n"
    "- toques: Customer touchpoints\n"
    "- overlap_inicial, asignacion_final: Assignment overlaps and allocations\n"
    "- ds_testlab: Test lab assignments\n"
    "- escenario: Business rule scenarios\n"
    "- conteo: Record counts (number of customers)\n"
    "- linea_ofrecida: Credit lines offered\n"
    "- npv: Net Present Value metrics\n"
    "- rentabilidad: Profitability figures\n"
    "- rr: Response rates\n"
    "- campania: Campaign identifiers\n"
    "- flag_declinado: Decline indicator (1=declined, 0=accepted)\n"
    "- causa_no_asignacion: Reasons for non-assignment\n\n"
    "INSTRUCTIONS:\n"
    "1. ALWAYS respond in English\n"
    "2. Base your answer ONLY on the data provided\n"
    "3. Be precise with numbers, use commas for thousands\n"
    "4. If you cannot answer from the data, say so clearly\n"
    "5. Provide insights and explain patterns when relevant\n"
    "6. Use structured formatting with bullet points\n"
)
