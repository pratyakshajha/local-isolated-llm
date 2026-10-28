#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PYTHON="${PYTHON:-python3}"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

if [[ "$(uname -s)" == "Darwin" ]]; then
  # requirements.txt selects the CUDA 13 wheel; use the platform wheel on macOS.
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -e "$ROOT"
else
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"
  "$VENV/bin/python" -m pip install -e "$ROOT" --no-deps
fi

"$VENV/bin/python" -m pip install httpx2==2.16.0 ruff==0.12.12
"$VENV/bin/python" -m pip check

cd "$ROOT/frontend"
npm ci
npm run build

printf '\nSetup complete. Run ./scripts/run.sh to start Wonderdesk.\n'
