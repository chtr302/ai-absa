"""ONNX Runtime helpers configured for CPU-only 2-core inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


RUNTIME_NAME = "ONNX CPU 2-Core"


def create_onnx_session(model_path: str):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = int(os.getenv("ORT_INTRA_OP_NUM_THREADS", "2"))
    sess_options.inter_op_num_threads = int(os.getenv("ORT_INTER_OP_NUM_THREADS", "1"))
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def get_model_size_mb(model_path: str) -> float:
    return round(Path(model_path).stat().st_size / (1024 * 1024), 3)


def _parse_json_output(value: Any) -> dict[str, Any] | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    if not isinstance(value, str):
        return None

    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None


def _normalize_quads(payload: dict[str, Any]) -> list[dict[str, str]]:
    quads = payload.get("quads") or []
    if not isinstance(quads, list):
        return []

    normalized = []
    for quad in quads:
        if not isinstance(quad, dict):
            continue
        normalized.append(
            {
                "aspect": str(quad.get("aspect") or ""),
                "opinion": str(quad.get("opinion") or ""),
                "category": str(quad.get("category") or ""),
                "sentiment": str(quad.get("sentiment") or ""),
            }
        )
    return normalized


def predict_one(session: Any, sentence: str) -> dict[str, Any]:
    inputs = session.get_inputs()
    if len(inputs) != 1 or "string" not in inputs[0].type.lower():
        input_signature = [
            {"name": item.name, "type": item.type, "shape": item.shape}
            for item in inputs
        ]
        raise ValueError(
            "This runtime-only Docker app expects an ONNX model with one string input "
            f"that returns JSON containing a 'quads' list. Model inputs: {input_signature}"
        )

    input_name = inputs[0].name
    outputs = session.run(None, {input_name: np.array([sentence], dtype=object)})

    for output in outputs:
        payload = _parse_json_output(output)
        if payload is not None:
            return {
                "sentence": sentence,
                "quads": _normalize_quads(payload),
            }

    output_signature = [
        {"name": item.name, "type": item.type, "shape": item.shape}
        for item in session.get_outputs()
    ]
    raise ValueError(
        "Model ran, but no JSON output with a 'quads' list was found. "
        f"Model outputs: {output_signature}"
    )
