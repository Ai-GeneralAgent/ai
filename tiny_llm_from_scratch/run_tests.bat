@echo off
setlocal
cd /d "%~dp0"
title Modern Tiny LLM - Automated System Tests
echo ===================================================
echo Running 24 Mandatory System Tests...
echo ===================================================
if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" tests\test_system.py
) else (
    python tests\test_system.py
)
pause

