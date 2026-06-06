FROM python:3.13-slim-bookworm

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

# Application source (always from project root, not the transfer snapshot)
COPY server.py store.py chunkers.py dashboard.py reset-store.sh /app/
RUN chmod +x /app/reset-store.sh

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
