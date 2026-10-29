@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
set /p PROMPT=Enter a starting prompt:
python generate.py --prompt "%PROMPT%" --tokens 250
pause
