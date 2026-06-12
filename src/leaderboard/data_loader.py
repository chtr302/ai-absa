"""Dataset loading utilities for the AI-ABSA dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "final_data" / "train_final.jsonl",
    PROJECT_ROOT / "data" / "processed" / "final_data" / "dashboard_pool.jsonl",
    PROJECT_ROOT / "data" / "train_final.jsonl",
    PROJECT_ROOT / "train_final.jsonl",
]
DEFAULT_DATASET_FILENAMES = [
    "train_final.jsonl",
    "dashboard_pool.jsonl",
    "test.jsonl",
    "val.jsonl",
]


def resolve_dataset_path(path: str | None = None) -> str:
    """Resolve a dataset path from explicit input, env var, or known defaults."""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))

    env_path = os.getenv("AI_ABSA_DATA_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(DEFAULT_DATASET_CANDIDATES)

    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if resolved.is_dir():
            for filename in DEFAULT_DATASET_FILENAMES:
                nested = resolved / filename
                if nested.exists():
                    return str(nested)
        if resolved.is_file():
            return str(resolved)

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find AI-ABSA dataset. Searched: {searched}")


def normalize_quad(q: dict[str, Any]) -> dict[str, str]:
    """Return a safe quad with all expected fields present as strings."""
    return {
        "aspect": str(q.get("aspect") or ""),
        "opinion": str(q.get("opinion") or ""),
        "category": str(q.get("category") or ""),
        "sentiment": str(q.get("sentiment") or ""),
    }


def load_dataset(path: str | None = None) -> list[dict[str, Any]]:
    """Load JSONL rows while skipping malformed lines and normalizing records."""
    dataset_path = resolve_dataset_path(path)
    rows: list[dict[str, Any]] = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(raw, dict):
                continue

            raw_quads = raw.get("quads") or []
            if not isinstance(raw_quads, list):
                raw_quads = []

            rows.append(
                {
                    "id": line_index,
                    "sentence": str(raw.get("sentence") or ""),
                    "quads": [
                        normalize_quad(q)
                        for q in raw_quads
                        if isinstance(q, dict)
                    ],
                }
            )

    return rows
