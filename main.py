import json
import pathlib

from src.data.data_loader import load_labeled, balance, split
from src.data.preprocess import preprocess
from src.models.model import build, fit, predict
from src.evaluation.evaluate import metrics, to_batch

# ── config ────────────────────────────────────────────────
DATA_PATH      = "data/r_LocalLlama_comments.jsonl"
RESULTS_DIR    = pathlib.Path("results")
MODEL_KIND     = "nb"     # "nb" | "lr"
N_PER_CLASS    = 33_000   # cap per sentiment class before splitting
# ──────────────────────────────────────────────────────────


def _texts(records: list[dict]) -> list[str]:
    return [preprocess(r["text"]) for r in records]

def _labels(records: list[dict]) -> list[str]:
    return [r["label"] for r in records]


def main() -> None:
    records = load_labeled(DATA_PATH)
    records = balance(records, n_per_class=N_PER_CLASS)

    train_data, val_data, test_data, lb_data = split(records)

    pipeline = build(kind=MODEL_KIND)
    fit(pipeline, _texts(train_data), _labels(train_data))

    # ── internal test set evaluation ──────────────────────
    test_texts  = _texts(test_data)
    test_labels = _labels(test_data)
    preds       = predict(pipeline, test_texts)

    acc, rep = metrics(test_labels, preds)
    print(f"=== Test Set  (n={len(test_data)}) ===")
    print(f"Accuracy : {acc:.4f}\n")
    print(rep)

    # ── leaderboard evaluation ────────────────────────────
    lb_texts  = _texts(lb_data)
    lb_labels = _labels(lb_data)
    lb_preds  = predict(pipeline, lb_texts)

    lb_acc, lb_rep = metrics(lb_labels, lb_preds)
    print(f"=== Leaderboard  (n={len(lb_data)}) ===")
    print(f"Accuracy : {lb_acc:.4f}\n")
    print(lb_rep)

    # ── save Pydantic output ──────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)

    batch_test = to_batch(test_texts, preds)
    (RESULTS_DIR / "test_predictions.json").write_text(
        batch_test.model_dump_json(indent=2), encoding="utf-8"
    )

    batch_lb = to_batch(lb_texts, lb_preds)
    (RESULTS_DIR / "leaderboard_predictions.json").write_text(
        batch_lb.model_dump_json(indent=2), encoding="utf-8"
    )

    # summary json for both splits
    summary = {
        "model": MODEL_KIND,
        "n_per_class_cap": N_PER_CLASS,
        "splits": {
            "train": len(train_data),
            "val":   len(val_data),
            "test":  len(test_data),
            "leaderboard": len(lb_data),
        },
        "test_accuracy":        round(acc, 4),
        "leaderboard_accuracy": round(lb_acc, 4),
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
