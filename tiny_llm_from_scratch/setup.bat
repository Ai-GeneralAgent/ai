@echo off
setlocal
cd /d "%~dp0"
echo Creating Python virtual environment...
py -3 -m venv .venv
if errorlevel 1 (
  echo Could not create environment. Install 64-bit Python 3.11 or 3.12 first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo.
echo Installing PyTorch CPU build for the easiest first setup...
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
  echo PyTorch install failed. See https://pytorch.org/get-started/locally/
  pause
  exit /b 1
)
echo.
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo.
echo Setup complete. Run train.bat to start training.
pause
