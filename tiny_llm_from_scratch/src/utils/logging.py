"""
Structured Logging & Console Formatting
"""
import sys
import time
from typing import Optional


class ANSIColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"


def configure_utf8_console():
    """Safe UTF-8 reconfiguration on Windows without re-wrapping buffers."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def log_step(step: int, total_steps: int, loss: float, lr: float, speed: float):
    msg = f"Step {step:5d}/{total_steps} | Loss: {loss:.4f} | LR: {lr:.2e} | {speed:.0f} tok/s"
    print(msg)

