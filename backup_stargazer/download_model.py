#!/usr/bin/env python3
'''
Download Gemma 3 Model for Offline Use

This script downloads the google/gemma-3-1b-it model from HuggingFace
and saves it to a local directory so the chatbot can run fully offline.

Usage:
    # Set your HuggingFace token first
    export HF_TOKEN="hf_your_token_here"

    # Run the download script
    python download_model.py

    # Or specify a custom output directory
    python download_model.py --output /path/to/model/directory

Note: You need to accept the Gemma license on HuggingFace first:
    https://huggingface.co/google/gemma-3-1b-it

Authors: Ivan Dario Penaloza Rojas <ip70574@citi.com>
Manager: Ivan Dario Penaloza Rojas <ip70574@citi.com>
'''

import os
import sys
import argparse


def download_gemma_model(
    model_id: str = "google/gemma-3-1b-it",
    output_dir: str = "/home/ivan/ProjectPrometheus/models/gemma-3-1b-it",
    token: str = None
):
    """
    Download the Gemma 3 model and tokenizer to a local directory.

    Args:
        model_id: HuggingFace model identifier
        output_dir: Local directory to save the model
        token: HuggingFace API token (required for gated models)
    """
    # Resolve token
    hf_token = token or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("=" * 60)
        print("ERROR: HuggingFace token is required.")
        print()
        print("Gemma is a gated model. You need to:")
        print("  1. Create an account at https://huggingface.co")
        print("  2. Accept the license at:")
        print(f"     https://huggingface.co/{model_id}")
        print("  3. Create a token at:")
        print("     https://huggingface.co/settings/tokens")
        print("  4. Set it: export HF_TOKEN='hf_your_token'")
        print("=" * 60)
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"  Gemma 3 Offline Model Downloader")
    print(f"{'=' * 60}")
    print(f"  Model:  {model_id}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Import transformers
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("ERROR: 'transformers' package not installed.")
        print("Run: pip install transformers torch")
        sys.exit(1)

    # Download tokenizer
    print("[1/2] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_dir)
    print(f"       Tokenizer saved to {output_dir}")

    # Download model
    print("[2/2] Downloading model (this may take several minutes)...")
    print("       Model size: ~2.5 GB for gemma-3-1b-it")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    model.save_pretrained(output_dir)
    print(f"       Model saved to {output_dir}")

    # Verify
    print()
    print(f"{'=' * 60}")
    print(f"  Download complete!")
    print(f"{'=' * 60}")
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, filenames in os.walk(output_dir)
        for f in filenames
    )
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    print(f"  Files:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            sz = os.path.getsize(fpath) / (1024**2)
            print(f"    {f} ({sz:.1f} MB)")
    print()
    print(f"  Set this path in your config or environment:")
    print(f"    export GEMMA_MODEL_PATH=\"{output_dir}\"")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Gemma 3 model for offline chatbot use"
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-1b-it",
        help="HuggingFace model ID (default: google/gemma-3-1b-it)"
    )
    parser.add_argument(
        "--output",
        default="/home/ivan/ProjectPrometheus/models/gemma-3-1b-it",
        help="Local directory to save the model"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)"
    )

    args = parser.parse_args()
    download_gemma_model(
        model_id=args.model,
        output_dir=args.output,
        token=args.token
    )
