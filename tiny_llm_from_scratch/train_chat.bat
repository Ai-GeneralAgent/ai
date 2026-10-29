@echo off
cd /d "%~dp0"
title TinyGPT Mixed Data Training

echo ==========================================
echo       TinyGPT Mixed Data Training
echo ==========================================
echo.

if not exist "finetune_chat.py" (
    echo ERROR: finetune_chat.py was not found.
    pause
    exit /b 1
)

if not exist "qa_seed.txt" (
    echo ERROR: qa_seed.txt was not found.
    pause
    exit /b 1
)

if not exist "data_merged.txt" (
    echo ERROR: data_merged.txt was not found.
    pause
    exit /b 1
)

if not exist "checkpoints\tiny_gpt_vocab_test.pt" (
    echo ERROR: The test checkpoint was not found.
    pause
    exit /b 1
)

if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON=..\.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" ".\finetune_chat.py" --checkpoint "checkpoints\tiny_gpt_vocab_test.pt" --examples "qa_seed.txt" --base-data "data_merged.txt" --steps 600 --batch-size 8 --eval-interval 100 --qa-ratio 0.70

echo.
echo Training process finished. Check the messages above for errors.
pause
