#!/usr/bin/env bash
# Run training on Linux / macOS / Colab
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "../.venv/bin/python" ]; then
    PYTHON_CMD="../.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "==================================================="
echo "Training Modern Transformer (v2.0.0)"
echo "==================================================="
$PYTHON_CMD train_modern.py "$@"

