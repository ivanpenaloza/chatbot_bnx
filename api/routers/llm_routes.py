'''
LLM Router - AI Chatbot with Multi-Model Offline Support

This module provides REST API endpoints for an AI-powered chatbot
using locally-stored LLM models (fully offline, no internet needed).

Supported models:
  - Gemma-3-1B-IT            (compact, ~3.8GB)
  - Meta-Llama-3.1-8B-Instruct (high quality, ~16GB)

Only one model is loaded in memory at a time. Switching models
unloads the current one first to conserve GPU/CPU resources.
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
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
            except ValueError as tok_err:
                if "sentencepiece" in str(tok_err).lower():
                    # Fallback: force fast tokenizer (uses tokenizer.json)
                    logger.warning(
                        "Slow tokenizer failed (sentencepiece issue), "
                        "falling back to fast tokenizer..."
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        local_files_only=True,
                        trust_remote_code=True,
                        use_fast=True,
                    )
                else:
                    raise

            # Load model
            logger.info("Loading model weights...")
            if self.device == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32

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

        When data_context is empty (no documents), the system prompt
        alone defines the assistant's behavior (identity mode).
        """
        # Build system content — only include DATA CONTEXT block if non-empty
        if data_context.strip():
            system_content = (
                f"{system_prompt}\n\n"
                f"DATA CONTEXT:\n{data_context}"
            )
        else:
            system_content = system_prompt

        user_content = user_message

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
        dynamic_context: str = "",
        system_prompt_override: str = "",
    ) -> tuple:
        """
        Generate a response using the currently loaded model.
        Returns (response_text, inference_time_seconds).

        system_prompt_override: if provided, replaces the default
        CHATBOT_SYSTEM_PROMPT (used by RAG flow routing).
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

        # Use override prompt if provided, otherwise default
        active_prompt = system_prompt_override if system_prompt_override else CHATBOT_SYSTEM_PROMPT

        try:
            input_text = self._build_prompt_text(
                system_prompt=active_prompt,
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

            # Cap max tokens to prevent runaway generation
            max_tokens = min(max(CHATBOT_MAX_NEW_TOKENS, 512), 1024)

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

    # Generate response
    answer, inference_time = model_manager.generate_response(
        user_message=chat.message,
        data_context="",
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
        "device": model_manager.device,
    })



