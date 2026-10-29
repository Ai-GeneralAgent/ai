@echo off
setlocal
cd /d "%~dp0"
title Modern Tiny LLM AI Assistant
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" chat.py %*
) else (
    python chat.py %*
)
pause
