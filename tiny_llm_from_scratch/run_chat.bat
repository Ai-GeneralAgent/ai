@echo off
setlocal
cd /d "%~dp0"
title Modern Tiny LLM AI Assistant - Chat
echo ===================================================
echo Starting Modern Tiny LLM Assistant...
echo ===================================================
if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" src\ui\cli.py %*
) else (
    python src\ui\cli.py %*
)
pause

