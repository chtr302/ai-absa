FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    flask \
    uvicorn \
    a2wsgi \
    onnxruntime \
    transformers

COPY src/leaderboard ./src/leaderboard
COPY src/models/basic ./src/models/basic
COPY src/models/__init__.py ./src/models/__init__.py
COPY src/__init__.py ./src/__init__.py
COPY templates ./templates
COPY static ./static

COPY models/advanced ./models/advanced
COPY models/basic ./models/basic

COPY data/processed/final_data/dashboard_pool.jsonl ./data/processed/final_data/dashboard_pool.jsonl
COPY data/processed/final_data/train_final.jsonl ./data/processed/final_data/train_final.jsonl
COPY data/processed/ontology/domain_ontology.json ./data/processed/ontology/domain_ontology.json

RUN mkdir -p /app/models/advanced /app/models/basic /app/data/processed/final_data /app/.cache/huggingface \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')"

EXPOSE 8003

CMD ["uvicorn", "src.leaderboard.app:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
