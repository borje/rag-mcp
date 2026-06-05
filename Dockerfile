FROM python:3.13-slim-bookworm

WORKDIR /app

# Install from pre-built offline wheel bundle (no internet needed)
COPY transfer/wheels /app/wheels
COPY transfer/src/requirements.frozen.txt /app/

RUN pip install --no-cache-dir --no-index \
      --find-links=/app/wheels \
      -r /app/requirements.frozen.txt \
    && rm -rf /app/wheels

# Pre-downloaded ONNX embedding model
COPY transfer/models /app/models

# Application source (always from project root, not the transfer snapshot)
COPY server.py store.py chunkers.py dashboard.py /app/

VOLUME /data

ENV RAG_MCP_DATA=/store
ENV FASTEMBED_CACHE_PATH=/app/models
ENV MCP_TRANSPORT=streamable-http
ENV FASTMCP_HOST=0.0.0.0
ENV FASTMCP_PORT=8000
ENV BASE_URL=http://localhost:8000
ENV FILES_ROOT=/data

EXPOSE 8000

CMD ["python", "/app/server.py"]
