@echo off
setlocal
cd /d "%~dp0"
title Modern Tiny LLM - Train from Scratch
echo ===================================================
echo Training Modern Transformer (v2.0.0)
echo ===================================================
if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" train_modern.py %*
) else (
    python train_modern.py %*
)
pause
