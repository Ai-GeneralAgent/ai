@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m unittest -v test_model.py
python train.py --steps 50 --eval-interval 25 --block-size 64 --n-embd 64 --n-head 4 --n-layer 2 --batch-size 8
pause
