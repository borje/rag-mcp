#!/usr/bin/env bash
# Reset persisted rag-mcp store files. Run inside the container, then restart it.
set -euo pipefail

STORE_DIR="${RAG_MCP_DATA:-$HOME/.local/share/rag-mcp}"

echo "==> Resetting rag-mcp store at $STORE_DIR"
mkdir -p "$STORE_DIR"
rm -f \
  "$STORE_DIR/meta.json" \
  "$STORE_DIR/vectors.npy" \
  "$STORE_DIR/mtimes.json" \
  "$STORE_DIR/bodies.json" \
  "$STORE_DIR/manifest.json"

echo "    Store files removed."
echo ""
echo "Next step: restart the container so startup ingest rebuilds the store."
echo "Example from host: docker compose restart rag-mcp"
