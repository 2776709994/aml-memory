FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AML_DATA_DIR=/data

# A Chinese VPS cannot reach pypi.org reliably; the mirror is a build argument so
# other users can override it (`docker build --build-arg PIP_INDEX_URL=...`).
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -i "$PIP_INDEX_URL" -r requirements.txt

COPY aml_memory ./aml_memory
# Operator tooling ships inside the image so the deployment self-check
# (`docker compose exec aml-memory python scripts/verify_endpoint.py ...`) needs
# nothing installed on the host.
COPY scripts ./scripts

RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# One worker on purpose: the per-user index and the SQLite file are process-local,
# and search latency is dominated by the embedder/reranker HTTP calls anyway.
CMD ["python", "-m", "uvicorn", "aml_memory.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "75"]
