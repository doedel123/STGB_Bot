#!/bin/sh
set -eu

if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

exec chainlit run app.py --host 0.0.0.0 --port "${PORT:-8000}"
