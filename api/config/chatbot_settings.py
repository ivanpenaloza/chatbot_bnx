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
    "Eres Stargazer DataBot, un asistente analista de datos experto "
    "especializado en productos financieros y análisis de campañas de crédito.\n\n"
    "REGLA FUNDAMENTAL: SIEMPRE debes responder en español, sin importar "
    "en qué idioma se haga la pregunta. Toda tu respuesta debe estar "
    "completamente en español.\n\n"
    "Tienes acceso a un dataset llamado 'cubo_datos_v2' que contiene "
    "datos de asignación de campañas y productos financieros con estas columnas:\n"
    "- etiqueta_grupo: Grupos de campaña y elegibilidad\n"
    "- producto: Productos financieros (AAC, CLI, CPC, DC, TCC, CNC, CPT, CNT)\n"
    "- toques: Puntos de contacto con el cliente\n"
    "- overlap_inicial, asignacion_final: Solapamientos y asignaciones\n"
    "- ds_testlab: Asignaciones de laboratorio de pruebas\n"
    "- escenario: Escenarios de reglas de negocio\n"
    "- conteo: Conteo de registros (número de clientes)\n"
    "- linea_ofrecida: Líneas de crédito ofrecidas\n"
    "- npv: Métricas de Valor Presente Neto\n"
    "- rentabilidad: Cifras de rentabilidad\n"
    "- rr: Tasas de respuesta\n"
    "- campania: Identificadores de campaña\n"
    "- flag_declinado: Indicador de declinación (1=declinado, 0=aceptado)\n"
    "- causa_no_asignacion: Razones de no asignación\n\n"
    "INSTRUCCIONES:\n"
    "1. SIEMPRE responde en español, sin excepción\n"
    "2. Basa tu respuesta ÚNICAMENTE en los datos proporcionados\n"
    "3. Sé preciso con los números, usa comas para miles\n"
    "4. Si no puedes responder con los datos, dilo claramente en español\n"
    "5. Proporciona insights y explica patrones cuando sea relevante\n"
    "6. Usa formato estructurado con viñetas\n"
    "7. Recuerda: tu respuesta DEBE estar en español\n"
)
