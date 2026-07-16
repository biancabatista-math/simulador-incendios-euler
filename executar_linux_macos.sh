#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv .venv
fi

source ".venv/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
