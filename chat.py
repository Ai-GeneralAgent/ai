#!/usr/bin/env python3
"""
Modern Tiny LLM AI Assistant (v2.1.0)
Unified root entry-point for interactive terminal chat, tools, RAG, and reasoning.
"""
import os
import sys
from pathlib import Path

# OpenBLAS / MKL memory protection on Windows CPU
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure UTF-8 stdout on Windows safely without re-wrapping
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR / "tiny_llm_from_scratch"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.ui.cli import launch_cli

if __name__ == "__main__":
    launch_cli()
