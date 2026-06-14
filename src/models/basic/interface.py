import argparse
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import Lock
from typing import Any

from .sentiment_model import ABSASentimentModel


MODEL_PATH_ENV = "AI_ABSA_BASELINE_MODEL_PATH"
DEFAULT_MODEL_PATH = Path("models/baseline/model.json")
INTERFACE_VERSION = "1.0"


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    configured_path = model_path or os.getenv(MODEL_PATH_ENV) or DEFAULT_MODEL_PATH
    resolved = Path(configured_path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Baseline model artifact not found: {resolved}. "
            f"Set {MODEL_PATH_ENV} or pass model_path explicitly."
        )
    return resolved


class BaselineModelInterface:
    """Stable application boundary for Flask/Docker integration."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = resolve_model_path(model_path)
        self.model = ABSASentimentModel.from_artifact(self.model_path)

    def predict(self, payload: str | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(payload, str):
            text = payload
            parent_context = ""
            thread_title = ""
        elif isinstance(payload, Mapping):
            text = str(
                payload.get("text")
                or payload.get("sentence")
                or payload.get("raw_text")
                or ""
            )
            parent_context = str(payload.get("parent_context") or "")
            thread_title = str(payload.get("thread_title") or "")
        else:
            raise TypeError("payload must be a string or JSON object")
        if not text.strip():
            raise ValueError(
                "payload must contain non-empty 'text', 'sentence', or 'raw_text'"
            )
        return self.model.predict(
            text,
            parent_context=parent_context,
            thread_title=thread_title,
        )

    def predict_batch(
        self, payloads: Iterable[str | Mapping[str, Any]]
    ) -> dict[str, Any]:
        results = [self.predict(payload) for payload in payloads]
        return {
            "results": results,
            "total_count": len(results),
            "model_name": self.model.MODEL_NAME,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "interface_version": INTERFACE_VERSION,
            "model_name": self.model.MODEL_NAME,
            "model_path": str(self.model_path),
            "model_size_bytes": self.model_path.stat().st_size,
            "schema_version": self.model.artifact_metadata.get(
                "schema_version", 1
            ),
            "training_type": self.model.artifact_metadata.get(
                "training_type", "controlled_rule_fitting"
            ),
        }


def create_model_interface(
    model_path: str | Path | None = None,
) -> BaselineModelInterface:
    return BaselineModelInterface(model_path)


_singleton_lock = Lock()
_singleton: BaselineModelInterface | None = None
_singleton_path: Path | None = None


def get_model_interface(
    model_path: str | Path | None = None,
) -> BaselineModelInterface:
    """Return one loaded model instance per application process."""
    global _singleton, _singleton_path
    resolved = resolve_model_path(model_path)
    with _singleton_lock:
        if _singleton is None or _singleton_path != resolved:
            _singleton = BaselineModelInterface(resolved)
            _singleton_path = resolved
        return _singleton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline model inference.")
    parser.add_argument("--model-path")
    parser.add_argument("--text", required=True)
    parser.add_argument("--parent-context", default="")
    parser.add_argument("--thread-title", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interface = get_model_interface(args.model_path)
    output = interface.predict(
        {
            "text": args.text,
            "parent_context": args.parent_context,
            "thread_title": args.thread_title,
        }
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
