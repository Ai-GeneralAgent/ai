@echo off
setlocal
cd /d "%~dp0"
echo Starting CPU-friendly BPE Transformer training...
python train_bpe.py --data data_merged_updated.txt --tokenizer tokenizer_bpe.json --steps 1000 --eval-interval 100 --batch-size 4 --block-size 256 --n-embd 192 --n-head 4 --n-layer 3
if errorlevel 1 (
  echo Training failed. Read the error above.
  pause
  exit /b 1
)
echo Training finished.
pause
