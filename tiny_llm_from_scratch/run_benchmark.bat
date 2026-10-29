@echo off
setlocal
cd /d "%~dp0"
title Modern Tiny LLM - Empirical Model Benchmark
echo ===================================================
echo Running Benchmark Suite: Legacy vs Modern
echo ===================================================
if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" compare_models.py
) else (
    python compare_models.py
)
pause

