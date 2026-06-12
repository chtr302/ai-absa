"""Flask API and UI entrypoint for the AI-ABSA dataset dashboard."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from .data_loader import load_dataset
from .onnx_runner import RUNTIME_NAME, create_onnx_session, get_model_size_mb, predict_one
from .stats import (
    compute_backend_unknowns,
    compute_category_distribution,
    compute_category_sentiment_matrix,
    compute_factor_focus,
    compute_overview,
    compute_sentiment_distribution,
    compute_top_aspects,
    filter_samples,
    paginate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
MODEL_TYPES = ("baseline", "advanced")
DEFAULT_MODEL_PATHS = {
    "advanced": "models/advanced/model.onnx",
    "baseline": "models/baseline/model.onnx",
}

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)

try:
    DATASET_ROWS: list[dict[str, Any]] = load_dataset()
except FileNotFoundError:
    DATASET_ROWS = []

MODEL_SESSIONS: dict[str, Any] = {}
MODEL_PATHS: dict[str, Path] = {}


def _error(message: str, status_code: int = 400):
    response = jsonify({"error": message})
    response.status_code = status_code
    return response


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(value or default)
    except ValueError:
        return default


def _sample_payload(row: dict[str, Any]) -> dict[str, Any]:
    quads = row.get("quads", [])
    return {
        "id": row.get("id"),
        "sentence": row.get("sentence", ""),
        "quad_count": len(quads),
        "quads": quads,
    }


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_model_path(model_type: str) -> str:
    env_name = f"AI_ABSA_{model_type.upper()}_MODEL_PATH"
    return os.getenv(env_name) or DEFAULT_MODEL_PATHS[model_type]


def _get_model_session(model_type: str):
    model_path = _default_model_path(model_type)
    resolved_model_path = _resolve_model_path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"{model_type} model path does not exist: {model_path}")

    if model_type not in MODEL_SESSIONS or MODEL_PATHS.get(model_type) != resolved_model_path:
        MODEL_SESSIONS[model_type] = create_onnx_session(str(resolved_model_path))
        MODEL_PATHS[model_type] = resolved_model_path

    return MODEL_SESSIONS[model_type], resolved_model_path


def _run_prediction(model_type: str, sentence: str) -> dict[str, Any]:
    session, resolved_model_path = _get_model_session(model_type)
    result = predict_one(session, sentence)
    return {
        "model_type": model_type,
        "model_path": str(resolved_model_path),
        "runtime": RUNTIME_NAME,
        **result,
    }


def _model_payload(model_type: str) -> dict[str, Any] | None:
    resolved_model_path = _resolve_model_path(_default_model_path(model_type))
    if not resolved_model_path.exists():
        return None

    return {
        "model_name": resolved_model_path.stem,
        "model_type": model_type,
        "model_path": str(resolved_model_path),
        "model_size_mb": get_model_size_mb(str(resolved_model_path)),
        "runtime": RUNTIME_NAME,
        "cpu_profile": "2 CPU cores",
        "status": "ready",
    }


@app.get("/")
def index():
    return render_template("leaderboard.html")


@app.get("/api/dataset/overview")
def dataset_overview():
    return jsonify({**compute_overview(DATASET_ROWS), **compute_backend_unknowns(DATASET_ROWS)})


@app.get("/api/dataset/category-distribution")
def category_distribution():
    return jsonify(compute_category_distribution(DATASET_ROWS))


@app.get("/api/dataset/sentiment-distribution")
def sentiment_distribution():
    return jsonify(compute_sentiment_distribution(DATASET_ROWS))


@app.get("/api/dataset/top-aspects")
def top_aspects():
    limit = _to_int(request.args.get("limit"), 12)
    return jsonify(compute_top_aspects(DATASET_ROWS, limit=limit))


@app.get("/api/dataset/category-sentiment-matrix")
def category_sentiment_matrix():
    return jsonify(compute_category_sentiment_matrix(DATASET_ROWS))


@app.get("/api/dataset/factor-focus")
def factor_focus():
    main_limit = _to_int(request.args.get("main_limit"), 30)
    related_limit = _to_int(request.args.get("related_limit"), 8)
    return jsonify(
        compute_factor_focus(
            DATASET_ROWS,
            main_limit=main_limit,
            related_limit=related_limit,
        )
    )


@app.get("/api/dataset/samples")
def dataset_samples():
    page = _to_int(request.args.get("page"), 1)
    page_size = _to_int(request.args.get("page_size"), 20)
    filtered = filter_samples(
        DATASET_ROWS,
        search=request.args.get("search"),
        category=request.args.get("category"),
        sentiment=request.args.get("sentiment"),
        quad_type=request.args.get("quad_type", "all"),
    )
    page_data = paginate(filtered, page=page, page_size=page_size)
    page_data["items"] = [_sample_payload(row) for row in page_data["items"]]
    return jsonify(page_data)


@app.get("/api/model-comparison")
def model_comparison():
    return jsonify(
        {
            "baseline": _model_payload("baseline"),
            "advanced": _model_payload("advanced"),
            "improvement": None,
        }
    )


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    sentence = str(body.get("sentence") or "").strip()
    include_baseline = bool(body.get("include_baseline"))

    if not sentence:
        return _error("sentence is required")

    try:
        results = {"advanced": _run_prediction("advanced", sentence)}
        if include_baseline:
            results["baseline"] = _run_prediction("baseline", sentence)
    except ImportError as exc:
        return _error(f"Prediction dependency missing: {exc}", 500)
    except FileNotFoundError as exc:
        return _error(str(exc), 404)
    except Exception as exc:
        return _error(f"Prediction failed: {exc}", 500)

    return jsonify({"sentence": sentence, "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
