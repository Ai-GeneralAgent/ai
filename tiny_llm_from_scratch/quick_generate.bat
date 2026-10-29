@echo off
setlocal
cd /d "%~dp0"
python generate_bpe.py --checkpoint checkpoints\tiny_bpe_gpt.pt --prompt "Artificial intelligence" --tokens 120 --temperature 0.8 --top-k 40
pause
