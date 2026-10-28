#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Virtual environment missing. Run ./scripts/setup.sh first." >&2
  exit 1
fi

cd "$ROOT"
"$PYTHON" -m unittest discover -v
"$PYTHON" -m ruff check app tests
cd "$ROOT/frontend"
npm test
npm run typecheck
npm run build
npm audit --audit-level=high
