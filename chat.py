#!/usr/bin/env python3
"""
Modern Tiny LLM AI Assistant (v2.0.0)
Unified root entry-point for interactive terminal chat, tools, RAG, and reasoning.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR / "tiny_llm_from_scratch"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.ui.cli import launch_cli

if __name__ == "__main__":
    launch_cli()
