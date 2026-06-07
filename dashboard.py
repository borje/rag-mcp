"""Browser dashboard helpers for inspecting the local RAG store."""

import html
from pathlib import Path


def file_url(source: str, files_root: Path, base_url: str) -> str | None:
    try:
        rel = Path(source).relative_to(files_root)
        return f"{base_url}/files/{rel}"
    except ValueError:
        return None


def display_path(source: str, files_root: Path) -> str:
    try:
        return str(Path(source).relative_to(files_root))
    except ValueError:
        return source


def preview(text: str, length: int = 300) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= length:
        return collapsed
    return collapsed[: length - 1].rstrip() + "…"


def dashboard_data(store, files_root: Path, base_url: str) -> dict:
    """Return dashboard data grouped by source with chunk previews only."""
    bodies = store.load_bodies()
    files: dict[str, dict] = {}
    for i, chunk in enumerate(store._chunks):
        source = chunk["source"]
        if source not in files:
            files[source] = {
                "source": source,
                "path": display_path(source, files_root),
                "file_url": file_url(source, files_root, base_url),
                "chunk_count": 0,
                "chunks": [],
            }
        files[source]["chunk_count"] += 1
        files[source]["chunks"].append(
            {
                "id": chunk["id"],
                "doc_title": chunk.get("doc_title"),
                "chunk_type": chunk.get("chunk_type"),
                "title": chunk.get("title"),
                "preview": preview(bodies[i] if i < len(bodies) else ""),
            }
        )

    grouped = sorted(files.values(), key=lambda f: f["path"])
    stats = store.stats()
    return {
        "total_sources": stats["total_sources"],
        "total_chunks": stats["total_chunks"],
        "model": stats["model"],
        "store_dir": stats["store_dir"],
        "files_root": str(files_root),
        "files": grouped,
    }


def dashboard_chunk(store, files_root: Path, chunk_id: str) -> dict | None:
    for i, chunk in enumerate(store._chunks):
        if chunk["id"] == chunk_id:
            bodies = store.load_bodies()
            return {
                "id": chunk["id"],
                "title": chunk.get("title"),
                "chunk_type": chunk.get("chunk_type"),
                "source": chunk["source"],
                "path": display_path(chunk["source"], files_root),
                "body": bodies[i] if i < len(bodies) else "",
            }
    return None


def dashboard_html(base_url: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>rag-mcp Dashboard</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0b1020; color: #e5e7eb; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; padding: 28px; background: radial-gradient(circle at 20% 0%, rgba(59, 130, 246, .22), transparent 32%), radial-gradient(circle at 90% 10%, rgba(16, 185, 129, .16), transparent 30%), #0b1020; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    header {{ display: flex; justify-content: space-between; gap: 20px; align-items: start; margin-bottom: 20px; }}
    h1 {{ margin: 0; font-size: clamp(36px, 7vw, 70px); line-height: .95; letter-spacing: -.06em; }}
    .muted {{ color: #9ca3af; }}
    .pill {{ display: inline-flex; gap: 8px; align-items: center; padding: 10px 14px; border-radius: 999px; color: #a7f3d0; background: rgba(6, 78, 59, .35); border: 1px solid rgba(52, 211, 153, .4); font-weight: 800; }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; background: #34d399; box-shadow: 0 0 18px #34d399; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 16px; }}
    .card, details {{ border: 1px solid rgba(148, 163, 184, .18); background: rgba(15, 23, 42, .76); border-radius: 20px; box-shadow: 0 24px 80px rgba(0,0,0,.24); }}
    .card {{ padding: 18px; }}
    .label {{ color: #93c5fd; font-size: 12px; text-transform: uppercase; letter-spacing: .14em; font-weight: 800; }}
    .metric {{ margin-top: 8px; font-size: 34px; font-weight: 900; letter-spacing: -.04em; }}
    input {{ width: 100%; margin: 0 0 16px; padding: 14px 16px; color: #e5e7eb; background: rgba(2, 6, 23, .42); border: 1px solid rgba(148, 163, 184, .22); border-radius: 16px; font: inherit; }}
    .file {{ margin-bottom: 12px; overflow: hidden; }}
    summary {{ cursor: pointer; padding: 16px 18px; list-style: none; }}
    summary::-webkit-details-marker {{ display: none; }}
    .file-head {{ display: flex; justify-content: space-between; gap: 18px; align-items: center; }}
    .path {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; overflow-wrap: anywhere; }}
    .count {{ color: #bfdbfe; font-weight: 800; white-space: nowrap; }}
    .chunks {{ display: grid; gap: 10px; padding: 0 16px 16px; }}
    .chunk {{ background: rgba(2, 6, 23, .36); border-radius: 14px; }}
    .chunk summary {{ padding: 13px 14px; }}
    .chunk-title {{ font-weight: 800; }}
    .type {{ color: #a7f3d0; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .preview, pre {{ color: #d1d5db; line-height: 1.5; }}
    .preview {{ margin-top: 6px; }}
    pre {{ margin: 0; padding: 0 14px 14px; white-space: pre-wrap; overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; }}
    a {{ color: #93c5fd; }}
    @media (max-width: 800px) {{ body {{ padding: 16px; }} header, .file-head {{ display: grid; }} .cards {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <div><h1>rag-mcp</h1><p class="muted">Live dashboard at {html.escape(base_url)}/dashboard</p></div>
      <div class="pill"><span class="dot"></span>Running</div>
    </header>
    <section class="cards">
      <div class="card"><div class="label">Sources</div><div class="metric" id="sources">-</div></div>
      <div class="card"><div class="label">Chunks</div><div class="metric" id="chunks">-</div></div>
      <div class="card"><div class="label">Model</div><div class="muted" id="model">-</div></div>
      <div class="card"><div class="label">Files Root</div><div class="muted" id="filesRoot">-</div></div>
    </section>
    <input id="filter" type="search" placeholder="Filter files, chunk titles, types, previews...">
    <section id="files" aria-live="polite"></section>
  </main>
  <script>
    let dashboard = null;
    const els = {{ sources: document.getElementById('sources'), chunks: document.getElementById('chunks'), model: document.getElementById('model'), filesRoot: document.getElementById('filesRoot'), files: document.getElementById('files'), filter: document.getElementById('filter') }};

    function text(value) {{ return value == null ? '' : String(value); }}

    function chunkMatches(chunk, q) {{
      return [chunk.title, chunk.chunk_type, chunk.preview, chunk.id].some(v => text(v).toLowerCase().includes(q));
    }}

    function render() {{
      const q = els.filter.value.trim().toLowerCase();
      els.files.textContent = '';
      for (const file of dashboard.files) {{
        const chunks = q ? file.chunks.filter(chunk => chunkMatches(chunk, q)) : file.chunks;
        const fileMatch = [file.path, file.source].some(v => text(v).toLowerCase().includes(q));
        if (q && !fileMatch && chunks.length === 0) continue;

        const details = document.createElement('details');
        details.className = 'file';
        const summary = document.createElement('summary');
        summary.innerHTML = '<div class="file-head"><div><div class="path"></div><div class="muted"></div></div><div class="count"></div></div>';
        summary.querySelector('.path').textContent = file.path;
        summary.querySelector('.muted').textContent = file.source;
        summary.querySelector('.count').textContent = `${{file.chunk_count}} chunks`;
        details.append(summary);

        const wrap = document.createElement('div');
        wrap.className = 'chunks';
        if (file.file_url) {{
          const link = document.createElement('a');
          link.href = file.file_url;
          link.textContent = 'Open source file';
          wrap.append(link);
        }}
        for (const chunk of chunks) {{
          const c = document.createElement('details');
          c.className = 'chunk';
          c.dataset.chunkId = chunk.id;
          c.innerHTML = '<summary><div class="type"></div><div class="chunk-title"></div><div class="preview"></div></summary><pre hidden></pre>';
          c.querySelector('.type').textContent = chunk.chunk_type || 'chunk';
          c.querySelector('.chunk-title').textContent = chunk.title || chunk.id;
          c.querySelector('.preview').textContent = chunk.preview;
          c.addEventListener('toggle', async () => {{
            const pre = c.querySelector('pre');
            if (!c.open || pre.dataset.loaded) return;
            pre.hidden = false;
            pre.textContent = 'Loading full chunk...';
            const response = await fetch(`/dashboard/chunk/${{encodeURIComponent(chunk.id)}}`);
            const data = await response.json();
            pre.textContent = data.body || '';
            pre.dataset.loaded = '1';
          }});
          wrap.append(c);
        }}
        details.append(wrap);
        els.files.append(details);
      }}
    }}

    fetch('/dashboard/data').then(r => r.json()).then(data => {{
      dashboard = data;
      els.sources.textContent = data.total_sources;
      els.chunks.textContent = data.total_chunks;
      els.model.textContent = data.model;
      els.filesRoot.textContent = data.files_root;
      render();
    }}).catch(err => {{ els.files.textContent = `Failed to load dashboard data: ${{err}}`; }});
    els.filter.addEventListener('input', render);
  </script>
</body>
</html>"""
