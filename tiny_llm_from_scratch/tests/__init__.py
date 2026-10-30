"""
Unit test suite package initializer.
Ensures source and model modules are resolved in sys.path regardless of execution directory.
"""
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
ROOT_DIR = PROJECT_DIR.parent

for p in (PROJECT_DIR, ROOT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

