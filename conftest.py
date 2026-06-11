"""Repo-root conftest: makes `import store` work for tests/ and isolates
the store from the developer's real data directory.

store.py reads RAG_MCP_DATA at import time, so this must run before any
test module imports store/server — otherwise collecting the suite would
construct (and potentially migrate) the real store.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["RAG_MCP_DATA"] = tempfile.mkdtemp(prefix="rag-mcp-test-store-")
