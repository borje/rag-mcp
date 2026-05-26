#!/usr/bin/env bash
# Run on a CONNECTED Linux x86_64 machine to bundle rag-mcp for offline transfer.
# Creates ./transfer/ with wheels, model cache, source, and install.sh.
#
# Requirements on this machine: internet access, curl (uv is auto-installed if missing).
# Requirements on the target (Debian Bookworm / Yellow): Python 3.13, no internet needed.

set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$WORK_DIR/transfer"
UV="$HOME/.local/bin/uv"

echo "==> Ensuring uv is available..."
if ! command -v "$UV" &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo "    uv $("$UV" --version)"

echo "==> Ensuring Python 3.13 is available via uv..."
"$UV" python install 3.13 --quiet
PY313=$("$UV" run --python 3.13 python -c "import sys; print(sys.executable)")
echo "    $($PY313 --version) at $PY313"

echo "==> Creating build venv..."
rm -rf "$WORK_DIR/.venv-build"
"$PY313" -m venv "$WORK_DIR/.venv-build"
# shellcheck disable=SC1091
source "$WORK_DIR/.venv-build/bin/activate"
pip install --upgrade pip --quiet

echo "==> Installing all packages (resolves cp313 dep versions)..."
pip install --quiet -r "$WORK_DIR/requirements.txt"
echo "    Installed: $(pip list 2>/dev/null | wc -l) packages"

echo "==> Downloading cp313 wheels (native Linux x86_64, no cross-compile flags)..."
mkdir -p "$OUT_DIR/wheels"
pip freeze > "$WORK_DIR/.frozen.txt"
pip download \
  --quiet \
  -r "$WORK_DIR/.frozen.txt" \
  -d "$OUT_DIR/wheels" \
  --only-binary=:all:
echo "    $(ls "$OUT_DIR/wheels" | wc -l) wheels downloaded"

echo "==> Pre-downloading fastembed model (BAAI/bge-small-en-v1.5)..."
MODEL_CACHE="$WORK_DIR/.model-cache"
mkdir -p "$MODEL_CACHE"
FASTEMBED_CACHE_PATH="$MODEL_CACHE" python - <<'PYEOF'
import os
from fastembed import TextEmbedding
print("    Downloading model...")
list(TextEmbedding("BAAI/bge-small-en-v1.5").embed(["warmup"]))
print("    Model ready.")
PYEOF

echo "==> Copying model cache..."
mkdir -p "$OUT_DIR/models"
if [ -d "$MODEL_CACHE" ] && [ "$(ls -A "$MODEL_CACHE")" ]; then
  cp -r "$MODEL_CACHE/." "$OUT_DIR/models/"
  echo "    $(du -sh "$OUT_DIR/models" | cut -f1) model cache"
else
  echo "    WARNING: model cache empty at $MODEL_CACHE"
fi

echo "==> Copying source..."
mkdir -p "$OUT_DIR/src"
cp "$WORK_DIR"/{server.py,chunkers.py,store.py,requirements.txt} "$OUT_DIR/src/"
cp "$WORK_DIR/.frozen.txt" "$OUT_DIR/src/requirements.frozen.txt"

echo "==> Writing install.sh..."
cat > "$OUT_DIR/install.sh" << 'INSTALL_EOF'
#!/usr/bin/env bash
# Run on the OFFLINE (Yellow / Debian Bookworm) machine.
# Requires: Python 3.13 (e.g. python:3.13-slim-bookworm base image)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/rag-mcp"

echo "==> Installing to $INSTALL_DIR ..."
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cp -r "$SCRIPT_DIR/src/"* "$INSTALL_DIR/"

echo "==> Creating virtual environment..."
python3.13 -m venv "$INSTALL_DIR/.venv"
# shellcheck disable=SC1091
source "$INSTALL_DIR/.venv/bin/activate"

echo "==> Installing packages (offline)..."
pip install --quiet --no-index --find-links="$SCRIPT_DIR/wheels" -r "$INSTALL_DIR/requirements.frozen.txt"
echo "    Done"

echo "==> Installing model cache..."
MODEL_DEST="$INSTALL_DIR/models"
mkdir -p "$MODEL_DEST"
if [ "$(ls -A "$SCRIPT_DIR/models" 2>/dev/null)" ]; then
  cp -r "$SCRIPT_DIR/models/." "$MODEL_DEST/"
  echo "    Model cache installed to $MODEL_DEST"
else
  echo "    WARNING: model cache not found — first search will fail on offline machine"
fi

echo "==> Registering with Claude Code..."
PYTHON_BIN="$INSTALL_DIR/.venv/bin/python"
claude mcp remove rag-mcp 2>/dev/null || true
claude mcp add rag-mcp -s user \
  -e FASTEMBED_CACHE_PATH="$MODEL_DEST" \
  -e RAG_MCP_DATA="$HOME/.local/share/rag-mcp" \
  -- "$PYTHON_BIN" "$INSTALL_DIR/server.py"

echo ""
echo "Done! Verify with: claude mcp list"
echo ""
echo "Usage in Claude Code:"
echo "  ingest_file /path/to/api-spec.yaml"
echo "  ingest_directory /path/to/docs"
echo "  search 'how to authenticate'"
INSTALL_EOF
chmod +x "$OUT_DIR/install.sh"

echo ""
echo "====================================================="
echo " Ready to transfer: $OUT_DIR"
echo ""
ls -lh "$OUT_DIR"
echo ""
echo " Copy the 'transfer/' folder to the offline machine"
echo " then run:  bash install.sh"
echo "====================================================="
