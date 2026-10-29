@echo off
setlocal
cd /d "%~dp0"
title Modern Tiny LLM - Resume Training
echo ===================================================
echo Resuming Modern Transformer Training from Step 6800
echo ===================================================
if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" train_modern.py --resume checkpoints\modern_base_v2_best.pt --data data_knowledge_corpus.txt --steps 10000 --batch-size 4 --grad-accum 2 --run-name modern_base_v2 %*
) else (
    python train_modern.py --resume checkpoints\modern_base_v2_best.pt --data data_knowledge_corpus.txt --steps 10000 --batch-size 4 --grad-accum 2 --run-name modern_base_v2 %*
)
pause

