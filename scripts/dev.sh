#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" || ! -x "$ROOT/frontend/node_modules/.bin/vite" ]]; then
  echo "Setup is incomplete. Run ./scripts/setup.sh first." >&2
  exit 1
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

"$PYTHON" "$ROOT/llm_server.py" "$@" &
backend_pid=$!
(cd "$ROOT/frontend" && exec ./node_modules/.bin/vite) &
frontend_pid=$!

cleanup() {
  kill "$backend_pid" "$frontend_pid" 2>/dev/null || true
  wait "$backend_pid" "$frontend_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Backend: http://127.0.0.1:8080/  |  Vite: http://127.0.0.1:5173/\n'
status=0
while kill -0 "$backend_pid" 2>/dev/null && kill -0 "$frontend_pid" 2>/dev/null; do
  sleep 1
done
if ! kill -0 "$backend_pid" 2>/dev/null; then
  wait "$backend_pid" || status=$?
else
  wait "$frontend_pid" || status=$?
fi
exit "$status"
