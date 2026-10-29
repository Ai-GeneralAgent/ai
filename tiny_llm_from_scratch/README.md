# Tiny LLM From Scratch (Windows)

This is an educational, character-level GPT-style Transformer. Its weights are initialized randomly and trained locally using PyTorch. It does not load a pretrained model.

## Requirements
- 64-bit Windows 10/11
- 64-bit Python 3.11 or 3.12 recommended
- Internet for the first PyTorch installation
- A few GB of free disk space; CPU works but can be slow
- NVIDIA GPU is optional. `setup.bat` installs CPU-only PyTorch for reliable first setup.

## Quick start
1. Extract the ZIP into a normal folder.
2. Double-click `setup.bat` and wait for installation to finish.
3. Double-click `quick_test.bat` for a short smoke test and tiny training run.
4. Double-click `train.bat` for the default 2,000-step run.
5. Double-click `generate.bat` to generate text from your saved checkpoint.

Or use Command Prompt in this folder:
```bat
setup.bat
.venv\Scripts\activate
python -m unittest -v test_model.py
python train.py --steps 2000
python generate.py --prompt "The " --tokens 250
```

## Project files
- `model.py`: Transformer blocks, causal self-attention, and generation
- `train.py`: character tokenizer, train/validation split, loss calculation, optimization, checkpoint saving
- `generate.py`: load your trained weights and generate text
- `data.txt`: tiny starter dataset; replace/expand it with clean text you have permission to use
- `test_model.py`: basic shape, loss, and generation tests
- `checkpoints/tiny_gpt.pt`: created after training

## Improve the dataset
The included `data.txt` is intentionally tiny and only proves that the pipeline runs. It is not enough for useful language ability. Add a much larger plain-text corpus with consistent encoding and appropriate usage rights. Keep the text in `data.txt` or pass another file with `python train.py --data path\to\your_corpus.txt`.

## Adjust training
- Fast smoke test: `python train.py --steps 50 --eval-interval 25 --block-size 64 --n-embd 64 --n-head 4 --n-layer 2 --batch-size 8`
- Longer run: `python train.py --steps 5000`
- If CPU RAM or GPU VRAM is limited, reduce `--batch-size`, `--block-size`, `--n-embd`, or `--n-layer`.
- To use CUDA-enabled PyTorch, use the official selector at https://pytorch.org/get-started/locally/ and select Windows, Pip, Python, and the appropriate CUDA option. Replace the CPU-only torch installation in your virtual environment with the command provided there.

## What "from scratch" means here
The architecture is implemented by you in `model.py`; weights start from random initialization and are learned from your dataset. This is not a pretrained model. The tokenizer is deliberately character-based to keep the first project understandable. A useful production LLM needs vastly more high-quality data, compute, evaluation, and safety work.

## Troubleshooting
- `py` not recognized: install 64-bit Python and enable the Python Launcher / PATH options.
- `No module named torch`: run `setup.bat` again or activate `.venv` and install PyTorch.
- Training seems slow: CPU training can be slow; use `quick_test.bat` first.
- `Dataset too small`: add more text or lower `--block-size`.
