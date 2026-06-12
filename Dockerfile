FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN uv pip install --system --no-cache \
    flask \
    gunicorn \
    onnxruntime


FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    ORT_INTRA_OP_NUM_THREADS=2 \
    ORT_INTER_OP_NUM_THREADS=1 \
    AI_ABSA_DATA_PATH=/app/data/processed/final_data/dashboard_pool.jsonl \
    AI_ABSA_ADVANCED_MODEL_PATH=/app/models/advanced/model.onnx \
    AI_ABSA_BASELINE_MODEL_PATH=/app/models/baseline/model.onnx

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/leaderboard/__init__.py ./src/leaderboard/__init__.py
COPY src/leaderboard/app.py ./src/leaderboard/app.py
COPY src/leaderboard/data_loader.py ./src/leaderboard/data_loader.py
COPY src/leaderboard/onnx_runner.py ./src/leaderboard/onnx_runner.py
COPY src/leaderboard/stats.py ./src/leaderboard/stats.py
COPY templates ./templates
COPY static ./static
COPY models ./models
COPY data/processed/final_data/dashboard_pool.jsonl ./data/processed/final_data/dashboard_pool.jsonl

RUN mkdir -p /app/models/advanced /app/models/baseline /app/data/processed/final_data \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app/models

USER appuser

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "src.leaderboard.app:app"]
