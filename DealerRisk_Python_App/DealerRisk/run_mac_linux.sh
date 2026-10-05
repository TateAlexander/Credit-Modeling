#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
echo "Open http://127.0.0.1:8765 in your browser when the server says it is ready."
exec .venv/bin/python app.py
