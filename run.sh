#!/usr/bin/env bash
# One-shot launcher: create the venv, install deps, register the kernel, open JupyterLab.
# Usage:  bash run.sh
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
PYVER="3.11"
KERNEL_NAME="kcc2026"
KERNEL_DISPLAY="KCC2026 (.venv ${PYVER})"

# 1. Pick an installer: prefer uv (fast, can fetch Python ${PYVER}), fall back to python venv.
if command -v uv >/dev/null 2>&1; then
  echo ">> uv detected"
  [ -d "$VENV" ] || uv venv --python "$PYVER" "$VENV"
  uv pip install --python "$VENV/bin/python" -r requirements.txt
else
  echo ">> uv not found, falling back to python3 -m venv"
  PY="$(command -v python${PYVER} || command -v python3)"
  [ -d "$VENV" ] || "$PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -r requirements.txt
fi

# 2. Register the Jupyter kernel so the notebooks bind to this venv.
"$VENV/bin/python" -m ipykernel install --user \
  --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"

# 3. Load secrets from .env (the notebook also loads it via python-dotenv).
if [ -f .env ]; then
  set -a; . ./.env; set +a
  echo ">> loaded .env"
elif [ -z "${TENSORMESH_API_KEY:-}" ]; then
  echo ">> No .env found. Create it from the template and fill in your key:"
  echo "     cp .env.example .env"
  echo "   (the live-call cells will otherwise prompt you via getpass)"
fi

# 4. Launch JupyterLab with a fixed token.
JUPYTER_TOKEN="kcc2026-tutorial"
echo ">> launching JupyterLab (token: ${JUPYTER_TOKEN}) ..."
exec "$VENV/bin/jupyter" lab --ip=0.0.0.0 --IdentityProvider.token="$JUPYTER_TOKEN"
