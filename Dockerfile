FROM python:3.13-slim-bookworm

WORKDIR /app

# Pick up Debian security fixes newer than the base image
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

# setuptools ships with the base image but isn't needed at runtime (CVE-2025-47273)
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip uninstall -y setuptools

# Application source (always from project root, not the transfer snapshot)
COPY server.py store.py chunkers.py dashboard.py reset-store.sh /app/
RUN chmod +x /app/reset-store.sh

# ponytail: fixed uid 1000; match host uid via build arg if bind-mount perms clash
RUN useradd --uid 1000 --create-home app \
    && mkdir -p /data /store /models \
    && chown app:app /data /store /models
USER app

VOLUME /data
VOLUME /models

ENV RAG_MCP_DATA=/store
ENV FASTEMBED_CACHE_PATH=/models
ENV MCP_TRANSPORT=streamable-http
ENV FASTMCP_HOST=0.0.0.0
ENV FASTMCP_PORT=8000
ENV BASE_URL=http://localhost:8000
ENV FILES_ROOT=/data

EXPOSE 8000

CMD ["python", "/app/server.py"]
