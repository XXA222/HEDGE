#!/usr/bin/env bash
set -u
REPO_ROOT="${1:-$(pwd)}"
SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then PYTHON="${PYTHON_BIN:-python}"; fi
"$PYTHON" "$SUITE_ROOT/validate_suite.py" || exit $?
exec "$PYTHON" "$SUITE_ROOT/run_suite.py" --repo-root "$REPO_ROOT" --validation-mode strict-oos --budget balanced --device cpu --strategy-device cpu
