'''
LLM Router - AI Chatbot with Multi-Model Offline Support

This module provides REST API endpoints for an AI-powered chatbot
that answers statistical questions about the cubo_datos_v2 dataset
using locally-stored LLM models (fully offline, no internet needed).

Supported models:
  - Qwen2.5-0.5B-Instruct  (fastest, ~1GB)
  - TinyLlama-1.1B-Chat     (fast, ~2.2GB)
  - Gemma-3-1B-IT            (best quality, ~3.8GB)

Only one model is loaded in memory at a time. Switching models
unloads the current one first to conserve GPU/CPU resources.

Authors: Ivan Dario Penaloza Rojas <ip70574@citi.com>
Manager: Ivan Dario Penaloza Rojas <ip70574@citi.com>
'''

import os
import sys
import gc
import time
import logging
import traceback
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Add project root to sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_KEY,
    CHATBOT_MAX_NEW_TOKENS,
    CHATBOT_TEMPERATURE,
    CHATBOT_TOP_P,
    CHATBOT_TOP_K,
    CHATBOT_REPETITION_PENALTY,
    CHATBOT_DO_SAMPLE,
    CHATBOT_CSV_PATH,
    CHATBOT_MAX_CONTEXT_ROWS,
    CHATBOT_TOP_N_CATEGORIES,
    CHATBOT_SYSTEM_PROMPT,
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Router Setup ────────────────────────────────────────────────────────────
router = APIRouter(
    prefix="/api/v1",
    tags=["chatbot"]
)

templates = Jinja2Templates(directory='templates')


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    message: str
    model_key: Optional[str] = None
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    model_used: Optional[str] = None
    inference_time: Optional[float] = None
    data_used: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ModelSwitchRequest(BaseModel):
    model_key: str


# ─── Data Analyzer ───────────────────────────────────────────────────────────

class DataAnalyzer:
    """
    Manages the cubo_datos_v2 dataset and provides statistical
    summaries for LLM context building.
    """

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.summary_cache: Optional[str] = None
        self.column_info: Optional[Dict] = None
        self.is_loaded: bool = False

    def load_data(self, csv_path: str) -> bool:
        """Load the CSV dataset into memory."""
        try:
            full_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                csv_path
            )
            if not os.path.exists(full_path):
                full_path = csv_path
            if not os.path.exists(full_path):
                logger.error(f"CSV file not found: {full_path}")
                return False

            self.df = pd.read_csv(full_path, low_memory=False)
            logger.info(
                f"Dataset loaded: {self.df.shape[0]:,} rows, "
                f"{self.df.shape[1]} columns"
            )
            self._build_column_info()
            self._build_summary_cache()
            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            traceback.print_exc()
            return False

    def _build_column_info(self):
        """Pre-compute column metadata."""
        self.column_info = {}
        for col in self.df.columns:
            info = {
                'dtype': str(self.df[col].dtype),
                'non_null': int(self.df[col].notna().sum()),
                'null_count': int(self.df[col].isna().sum()),
                'unique': int(self.df[col].nunique()),
            }
            if self.df[col].dtype in ['int64', 'float64']:
                info['stats'] = {
                    'mean': round(float(self.df[col].mean()), 4),
                    'median': round(float(self.df[col].median()), 4),
                    'std': round(float(self.df[col].std()), 4),
                    'min': round(float(self.df[col].min()), 4),
                    'max': round(float(self.df[col].max()), 4),
                    'sum': round(float(self.df[col].sum()), 4),
                }
            else:
                top_values = (
                    self.df[col].value_counts()
                    .head(CHATBOT_TOP_N_CATEGORIES)
                )
                info['top_values'] = {
                    str(k): int(v) for k, v in top_values.items()
                }
            self.column_info[col] = info

    def _build_summary_cache(self):
        """Build comprehensive text summary for LLM context."""
        parts = []
        parts.append("DATASET SUMMARY: cubo_datos_v2.csv")
        parts.append(
            f"Total records: {len(self.df):,} | "
            f"Columns: {len(self.df.columns)}"
        )
        parts.append(
            f"Column names: {', '.join(self.df.columns.tolist())}"
        )
        parts.append("")

        # Categorical columns
        cat_cols = self.df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            parts.append("CATEGORICAL COLUMNS:")
            for col in cat_cols:
                vc = (
                    self.df[col].value_counts()
                    .head(CHATBOT_TOP_N_CATEGORIES)
                )
                vals_str = ", ".join(
                    [f"{k}({v:,})" for k, v in vc.items()]
                )
                parts.append(
                    f"  {col} [{self.df[col].nunique()} unique]: "
                    f"{vals_str}"
                )
            parts.append("")

        # Numerical columns
        num_cols = self.df.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            parts.append("NUMERICAL COLUMNS:")
            for col in num_cols:
                s = self.df[col].describe()
                parts.append(
                    f"  {col}: mean={s['mean']:,.4f}, "
                    f"median={s['50%']:,.4f}, "
                    f"min={s['min']:,.4f}, max={s['max']:,.4f}, "
                    f"sum={self.df[col].sum():,.4f}"
                )
            parts.append("")

        # Cross-tabulations
        if 'producto' in self.df.columns and 'conteo' in self.df.columns:
            parts.append("PRODUCT SUMMARY (by conteo sum):")
            prod_summary = (
                self.df.groupby('producto')['conteo']
                .agg(['sum', 'mean', 'count'])
                .sort_values('sum', ascending=False)
            )
            for prod, row in prod_summary.iterrows():
                parts.append(
                    f"  {prod}: total_conteo={row['sum']:,.0f}, "
                    f"avg={row['mean']:,.2f}, records={row['count']:,}"
                )
            parts.append("")

        if 'escenario' in self.df.columns and 'conteo' in self.df.columns:
            parts.append("SCENARIO SUMMARY:")
            esc_summary = (
                self.df.groupby('escenario')['conteo']
                .agg(['sum', 'count'])
                .sort_values('sum', ascending=False)
            )
            for esc, row in esc_summary.iterrows():
                parts.append(
                    f"  {esc}: total_conteo={row['sum']:,.0f}, "
                    f"records={row['count']:,}"
                )
            parts.append("")

        if 'flag_declinado' in self.df.columns:
            parts.append("DECLINE STATUS:")
            decline_counts = self.df['flag_declinado'].value_counts()
            total = len(self.df)
            for val, count in decline_counts.items():
                label = "Declined" if val == 1 else "Accepted"
                pct = (count / total) * 100
                parts.append(
                    f"  {label} (flag={val}): {count:,} ({pct:.1f}%)"
                )
            parts.append("")

        if 'campania' in self.df.columns and 'conteo' in self.df.columns:
            parts.append("CAMPAIGN SUMMARY:")
            camp_summary = (
                self.df.groupby('campania')['conteo']
                .agg(['sum', 'count'])
                .sort_values('sum', ascending=False)
            )
            for camp, row in camp_summary.iterrows():
                parts.append(
                    f"  {camp}: total_conteo={row['sum']:,.0f}, "
                    f"records={row['count']:,}"
                )
            parts.append("")

        # NPV and profitability by product
        if all(c in self.df.columns
               for c in ['producto', 'npv', 'rentabilidad']):
            parts.append("PROFITABILITY BY PRODUCT:")
            prof_summary = (
                self.df.groupby('producto')
                .agg({'npv': ['sum', 'mean'],
                      'rentabilidad': ['sum', 'mean']})
                .sort_values(('npv', 'sum'), ascending=False)
            )
            for prod in prof_summary.index:
                npv_sum = prof_summary.loc[prod, ('npv', 'sum')]
                npv_mean = prof_summary.loc[prod, ('npv', 'mean')]
                rent_sum = prof_summary.loc[prod, ('rentabilidad', 'sum')]
                parts.append(
                    f"  {prod}: npv_total={npv_sum:,.2f}, "
                    f"npv_avg={npv_mean:,.2f}, "
                    f"rent_total={rent_sum:,.2f}"
                )

        self.summary_cache = "\n".join(parts)
        logger.info(
            f"Summary cache built: {len(self.summary_cache)} chars"
        )

    def get_dynamic_context(self, question: str) -> str:
        """Generate additional context based on keywords in the question."""
        if self.df is None:
            return ""

        extra = []
        q = question.lower()

        # Product-specific
        if 'producto' in self.df.columns:
            for prod in self.df['producto'].unique():
                if prod.lower() in q:
                    sub = self.df[self.df['producto'] == prod]
                    extra.append(f"\nFILTERED for producto='{prod}':")
                    extra.append(f"  Records: {len(sub):,}")
                    for nc in sub.select_dtypes(include=['number']).columns:
                        extra.append(
                            f"  {nc}: sum={sub[nc].sum():,.4f}, "
                            f"mean={sub[nc].mean():,.4f}"
                        )

        # Decline analysis
        decline_kw = ['declin', 'rechaz', 'decline', 'reject']
        if (any(kw in q for kw in decline_kw)
                and 'flag_declinado' in self.df.columns):
            declined = self.df[self.df['flag_declinado'] == 1]
            accepted = self.df[self.df['flag_declinado'] == 0]
            extra.append("\nDECLINE ANALYSIS:")
            extra.append(f"  Declined: {len(declined):,}")
            extra.append(f"  Accepted: {len(accepted):,}")
            if 'causa_no_asignacion' in declined.columns:
                reasons = (
                    declined['causa_no_asignacion']
                    .value_counts().head(10)
                )
                extra.append("  Top decline reasons:")
                for reason, count in reasons.items():
                    extra.append(f"    - {reason}: {count:,}")

        # Ranking queries
        ranking_kw = ['top', 'mayor', 'mejor', 'highest', 'ranking']
        if any(kw in q for kw in ranking_kw):
            if ('producto' in self.df.columns
                    and 'conteo' in self.df.columns):
                top_prods = (
                    self.df.groupby('producto')['conteo']
                    .sum().sort_values(ascending=False)
                )
                extra.append("\nPRODUCT RANKING by total conteo:")
                for i, (prod, total) in enumerate(
                    top_prods.items(), 1
                ):
                    extra.append(f"  {i}. {prod}: {total:,.0f}")

        return "\n".join(extra) if extra else ""

    def get_data_info(self) -> Dict[str, Any]:
        """Return dataset metadata for the frontend."""
        if not self.is_loaded:
            return {"loaded": False}
        return {
            "loaded": True,
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": self.df.columns.tolist(),
            "memory_mb": round(
                self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2
            ),
        }


# ─── Initialize Data Analyzer ────────────────────────────────────────────────
data_analyzer = DataAnalyzer()


# ─── Multi-Model Manager ─────────────────────────────────────────────────────

class MultiModelManager:
    """
    Manages multiple offline LLM models. Only one model is loaded
    in memory at a time. Switching models unloads the current one
    first to conserve GPU/CPU resources.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.active_model_key: Optional[str] = None
        self.is_ready = False
        self.is_loading = False
        self.device = "cpu"

    def _unload_current(self):
        """Unload the current model from memory."""
        if self.model is not None:
            logger.info(
                f"Unloading model: {self.active_model_key}"
            )
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.is_ready = False
            self.active_model_key = None

            # Force garbage collection + clear CUDA cache
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared")
            except ImportError:
                pass

    def load_model(self, model_key: Optional[str] = None):
        """Load a model by key. Unloads any currently loaded model first."""
        if model_key is None:
            model_key = DEFAULT_MODEL_KEY

        if model_key not in AVAILABLE_MODELS:
            logger.error(f"Unknown model key: {model_key}")
            return False

        # Already loaded
        if self.active_model_key == model_key and self.is_ready:
            logger.info(f"Model already loaded: {model_key}")
            return True

        self.is_loading = True
        model_info = AVAILABLE_MODELS[model_key]
        model_path = model_info["local_dir"]

        if not os.path.isdir(model_path):
            logger.error(
                f"Model directory not found: {model_path}\n"
                f"Please run 'python download_model.py' first."
            )
            self.is_loading = False
            return False

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # Unload current model first
            self._unload_current()

            logger.info(
                f"Loading model '{model_key}' from: {model_path}"
            )

            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            logger.info(f"Using device: {self.device}")

            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )

            # Load model
            logger.info("Loading model weights...")
            dtype = (
                torch.float16 if self.device == "cuda"
                else torch.float32
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            self.active_model_key = model_key
            self.is_ready = True
            self.is_loading = False

            display = model_info["display_name"]
            logger.info(
                f"Model '{display}' loaded successfully (OFFLINE)"
            )
            return True

        except ImportError as e:
            logger.error(
                f"Missing dependency: {e}. "
                f"Run: pip install transformers torch"
            )
            self.is_loading = False
            return False
        except Exception as e:
            logger.error(f"Error loading model '{model_key}': {e}")
            traceback.print_exc()
            self.is_loading = False
            return False

    def _build_prompt_text(
        self,
        system_prompt: str,
        user_message: str,
        data_context: str,
    ) -> str:
        """
        Build the final prompt string using the model's chat template
        when available, with proper system/user role separation.
        Falls back to a manual template for models without one.
        """
        # Separate system and user roles so each model can format
        # them according to its own chat template conventions.
        system_content = (
            f"{system_prompt}\n\n"
            f"DATA CONTEXT:\n{data_context}"
        )
        user_content = (
            f"{user_message}\n\n"
            f"Provide a detailed and accurate answer in English "
            f"based on the data provided."
        )

        # Try system+user roles first (Qwen, Gemma support this)
        conversation_with_system = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        # Fallback: single user role (TinyLlama sometimes lacks system)
        conversation_single = [
            {"role": "user", "content": (
                f"{system_content}\n\n"
                f"USER QUESTION: {user_content}"
            )},
        ]

        if hasattr(self.tokenizer, 'apply_chat_template'):
            # First try with system role
            try:
                return self.tokenizer.apply_chat_template(
                    conversation_with_system,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
            # Fallback to single user role
            try:
                return self.tokenizer.apply_chat_template(
                    conversation_single,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Manual fallback for models without chat template
        return (
            f"### System:\n{system_content}\n\n"
            f"### User:\n{user_content}\n\n"
            f"### Assistant:\n"
        )

    def generate_response(
        self,
        user_message: str,
        data_context: str,
        dynamic_context: str = ""
    ) -> tuple:
        """
        Generate a response using the currently loaded model.
        Returns (response_text, inference_time_seconds).
        """
        if not self.is_ready:
            return (
                "No hay modelo cargado. Selecciona un modelo "
                "en la barra lateral.",
                0.0
            )

        import torch

        # Merge contexts
        context = data_context
        if dynamic_context:
            context += "\n\n" + dynamic_context

        try:
            input_text = self._build_prompt_text(
                system_prompt=CHATBOT_SYSTEM_PROMPT,
                user_message=user_message,
                data_context=context,
            )

            # Use a larger max_length so the full context + sample
            # answers are not truncated before the actual question.
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.device)

            # Allow longer answers (the sample answers are 300-600
            # tokens in Spanish).
            max_tokens = max(CHATBOT_MAX_NEW_TOKENS, 1024)

            t0 = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=CHATBOT_TEMPERATURE,
                    top_p=CHATBOT_TOP_P,
                    top_k=CHATBOT_TOP_K,
                    repetition_penalty=CHATBOT_REPETITION_PENALTY,
                    do_sample=CHATBOT_DO_SAMPLE,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            inference_time = round(time.time() - t0, 2)

            # Decode only the new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            if not response:
                response = (
                    "I was unable to generate a response. "
                    "Could you please rephrase your question?"
                )

            return (response, inference_time)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            traceback.print_exc()
            return (
                f"Error generating response: {str(e)}",
                0.0
            )

    def get_status(self) -> Dict[str, Any]:
        """Return the current model manager status."""
        models_status = []
        for key, info in AVAILABLE_MODELS.items():
            available = os.path.isdir(info["local_dir"])
            models_status.append({
                "key": key,
                "display_name": info["display_name"],
                "size": info["size"],
                "description": info["description"],
                "available": available,
                "active": key == self.active_model_key,
            })
        return {
            "active_model": self.active_model_key,
            "is_ready": self.is_ready,
            "is_loading": self.is_loading,
            "device": self.device,
            "models": models_status,
        }


# ─── Initialize Model Manager ────────────────────────────────────────────────
model_manager = MultiModelManager()


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    """Serve the chatbot HTML page."""
    return templates.TemplateResponse("chatbot.html", {
        "request": request,
    })


@router.post("/chatbot/ask", response_class=JSONResponse)
async def chatbot_ask(chat: ChatMessage):
    """Process a user message and return an AI-generated response."""
    if not chat.message.strip():
        raise HTTPException(
            status_code=400, detail="Message cannot be empty"
        )

    # Ensure data is loaded
    if not data_analyzer.is_loaded:
        data_analyzer.load_data(CHATBOT_CSV_PATH)

    if not data_analyzer.is_loaded:
        return JSONResponse(content={
            "response": (
                "Dataset not available. The file "
                "cubo_datos_v2.csv could not be loaded."
            ),
            "error": "data_not_loaded"
        })

    # Build context
    summary = data_analyzer.summary_cache or "No data summary available."
    dynamic = data_analyzer.get_dynamic_context(chat.message)

    # Generate response
    answer, inference_time = model_manager.generate_response(
        user_message=chat.message,
        data_context=summary,
        dynamic_context=dynamic,
    )

    active_info = AVAILABLE_MODELS.get(
        model_manager.active_model_key, {}
    )

    return JSONResponse(content={
        "response": answer,
        "model_used": active_info.get(
            "display_name", "Unknown"
        ),
        "model_key": model_manager.active_model_key,
        "inference_time": inference_time,
        "data_used": {
            "rows_analyzed": (
                len(data_analyzer.df)
                if data_analyzer.df is not None else 0
            ),
            "dynamic_context_used": bool(dynamic),
        }
    })


@router.get("/chatbot/models", response_class=JSONResponse)
async def chatbot_models():
    """Return available models and their status."""
    return JSONResponse(content=model_manager.get_status())


@router.post("/chatbot/models/switch", response_class=JSONResponse)
async def chatbot_switch_model(req: ModelSwitchRequest):
    """Switch the active model. Unloads the current one first."""
    if req.model_key not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {req.model_key}. "
                   f"Available: {list(AVAILABLE_MODELS.keys())}"
        )

    if model_manager.is_loading:
        raise HTTPException(
            status_code=409,
            detail="A model is currently being loaded. Please wait."
        )

    success = model_manager.load_model(req.model_key)

    if success:
        info = AVAILABLE_MODELS[req.model_key]
        return JSONResponse(content={
            "status": "success",
            "message": f"{info['display_name']} loaded successfully",
            "active_model": req.model_key,
            "display_name": info["display_name"],
        })
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {req.model_key}"
        )


@router.get("/chatbot/data-info", response_class=JSONResponse)
async def chatbot_data_info():
    """Return metadata about the loaded dataset."""
    if not data_analyzer.is_loaded:
        data_analyzer.load_data(CHATBOT_CSV_PATH)
    return JSONResponse(content=data_analyzer.get_data_info())


@router.get("/chatbot/health", response_class=JSONResponse)
async def chatbot_health():
    """Check chatbot service health."""
    active_info = AVAILABLE_MODELS.get(
        model_manager.active_model_key, {}
    )
    return JSONResponse(content={
        "status": "healthy",
        "active_model": model_manager.active_model_key,
        "model_display_name": active_info.get("display_name", "None"),
        "model_ready": model_manager.is_ready,
        "model_loading": model_manager.is_loading,
        "data_loaded": data_analyzer.is_loaded,
        "data_rows": (
            len(data_analyzer.df)
            if data_analyzer.df is not None else 0
        ),
        "device": model_manager.device,
    })


@router.get("/chatbot/quick-stats", response_class=JSONResponse)
async def chatbot_quick_stats():
    """Return quick statistical summaries for the frontend."""
    if not data_analyzer.is_loaded:
        data_analyzer.load_data(CHATBOT_CSV_PATH)
    if not data_analyzer.is_loaded:
        raise HTTPException(status_code=404, detail="Data not loaded")

    df = data_analyzer.df
    stats = {
        "total_records": len(df),
        "total_columns": len(df.columns),
    }
    if 'producto' in df.columns:
        stats["unique_products"] = int(df['producto'].nunique())
        stats["top_products"] = (
            df['producto'].value_counts().head(5).to_dict()
        )
    if 'conteo' in df.columns:
        stats["total_conteo"] = int(df['conteo'].sum())
    if 'npv' in df.columns:
        stats["total_npv"] = round(float(df['npv'].sum()), 2)
    if 'flag_declinado' in df.columns:
        total = len(df)
        declined = int((df['flag_declinado'] == 1).sum())
        stats["decline_rate"] = round((declined / total) * 100, 1)
    if 'campania' in df.columns:
        stats["unique_campaigns"] = int(df['campania'].nunique())
    if 'escenario' in df.columns:
        stats["unique_scenarios"] = int(df['escenario'].nunique())

    return JSONResponse(content=stats)
