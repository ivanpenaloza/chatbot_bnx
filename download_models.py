#!/usr/bin/env python3
"""
Satriani — Download All Models for Offline Use

Downloads chat and embedding models to /home/ivan/ProjectPrometheus/models/

Chat models (require HF_TOKEN for gated models):
  - Meta-Llama-3.1-8B-Instruct  (~16 GB)

Embedding models (no token needed):
  - gte-large-en-v1.5            (~670 MB)
  - gte-multilingual-base        (~560 MB)

Usage:
    export HF_TOKEN="hf_your_token_here"
    python download_models.py
    python download_models.py --only embeddings   # just embedding models
    python download_models.py --only chat          # just chat models
    python download_models.py --model llama        # specific model
"""

import os
import sys
import argparse

MODELS_DIR = "/home/ivan/ProjectPrometheus/models"

MODELS = {
    # Chat models
    "llama": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "local_dir": os.path.join(MODELS_DIR, "Meta-Llama-3.1-8B-Instruct"),
        "type": "chat",
        "gated": True,
        "size": "~16 GB",
    },
    # Embedding models
    "gte-large": {
        "hf_id": "Alibaba-NLP/gte-large-en-v1.5",
        "local_dir": os.path.join(MODELS_DIR, "gte-large-en-v1.5"),
        "type": "embedding",
        "gated": False,
        "size": "~670 MB",
    },
    "gte-multi": {
        "hf_id": "Alibaba-NLP/gte-multilingual-base",
        "local_dir": os.path.join(MODELS_DIR, "gte-multilingual-base"),
        "type": "embedding",
        "gated": False,
        "size": "~560 MB",
    },
}


def get_dir_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


def download_chat_model(model_id, output_dir, token):
    """Download a causal LM (chat model) using snapshot_download.
    This avoids loading the full model into RAM — just downloads files."""
    from huggingface_hub import snapshot_download

    os.makedirs(output_dir, exist_ok=True)

    print(f"  Downloading all model files...")
    snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        token=token if token else None,
        ignore_patterns=["consolidated.safetensors"],  # skip duplicate single-file
    )


def download_embedding_model(model_id, output_dir, token):
    """Download a sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer

    os.makedirs(output_dir, exist_ok=True)

    print(f"  Downloading embedding model...")
    model = SentenceTransformer(
        model_id,
        token=token if token else None,
        trust_remote_code=True,
    )
    model.save(output_dir)


def download_model(key, info, token):
    """Download a single model."""
    model_id = info["hf_id"]
    output_dir = info["local_dir"]
    model_type = info["type"]

    # Check if already downloaded
    if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 2:
        size_gb = get_dir_size(output_dir) / (1024 ** 3)
        print(f"  Already exists ({size_gb:.2f} GB). Skipping.")
        return True

    if info["gated"] and not token:
        print(f"  SKIPPED — gated model requires HF_TOKEN.")
        print(f"  Accept license at: https://huggingface.co/{model_id}")
        return False

    try:
        if model_type == "chat":
            download_chat_model(model_id, output_dir, token)
        else:
            download_embedding_model(model_id, output_dir, token)

        size_gb = get_dir_size(output_dir) / (1024 ** 3)
        print(f"  Done! Size: {size_gb:.2f} GB")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Satriani models")
    parser.add_argument("--only", choices=["chat", "embeddings"],
                        help="Download only chat or embedding models")
    parser.add_argument("--model", type=str,
                        help="Download a specific model by key: " +
                             ", ".join(MODELS.keys()))
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN", "")

    print("=" * 60)
    print("  Satriani — Model Downloader")
    print("=" * 60)
    print(f"  Models dir: {MODELS_DIR}")
    print(f"  HF Token:   {'set' if token else 'NOT SET'}")
    print("=" * 60)
    print()

    if not token:
        print("WARNING: HF_TOKEN not set. Gated models (Llama)")
        print("will be skipped. Set it with: export HF_TOKEN='hf_...'")
        print()

    # Filter models
    targets = MODELS
    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(MODELS.keys())}")
            sys.exit(1)
        targets = {args.model: MODELS[args.model]}
    elif args.only == "chat":
        targets = {k: v for k, v in MODELS.items() if v["type"] == "chat"}
    elif args.only == "embeddings":
        targets = {k: v for k, v in MODELS.items() if v["type"] == "embedding"}

    results = {}
    for key, info in targets.items():
        print(f"[{info['type'].upper()}] {info['hf_id']} ({info['size']})")
        print(f"  → {info['local_dir']}")
        ok = download_model(key, info, token)
        results[key] = ok
        print()

    # Summary
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    for key, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED/SKIPPED"
        print(f"  {key:20s} {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
