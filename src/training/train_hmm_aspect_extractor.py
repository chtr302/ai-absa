import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import BioSequenceSample, bio_tags_to_token_spans, load_bio_samples, split_samples
from src.models.hmm_aspect_extractor import HiddenMarkovAspectExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an HMM baseline for BIO aspect extraction.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/mendeley_bio_tagged.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hmm_aspect_extractor"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--min-token-freq", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--no-merge-reviews", action="store_true")
    parser.add_argument("--preview-limit", type=int, default=5)
    return parser.parse_args()


def spans_to_text(tokens: Sequence[str], spans: Sequence[Tuple[int, int]]) -> List[str]:
    return [" ".join(tokens[start:end]) for start, end in spans]


def evaluate_split(
    model: HiddenMarkovAspectExtractor,
    samples: Sequence[BioSequenceSample],
    preview_limit: int = 5,
) -> Dict[str, object]:
    total_tokens = 0
    correct_tokens = 0
    exact_sequence_matches = 0
    gold_span_total = 0
    predicted_span_total = 0
    true_positive_spans = 0
    previews: List[Dict[str, object]] = []

    for sample in samples:
        predicted_tags = model.predict(sample.tokens)
        total_tokens += len(sample.tokens)
        correct_tokens += sum(
            1 for gold_tag, predicted_tag in zip(sample.bio_tags, predicted_tags) if gold_tag == predicted_tag
        )
        if predicted_tags == sample.bio_tags:
            exact_sequence_matches += 1

        gold_spans = set(bio_tags_to_token_spans(sample.bio_tags))
        predicted_spans = set(bio_tags_to_token_spans(predicted_tags))
        gold_span_total += len(gold_spans)
        predicted_span_total += len(predicted_spans)
        true_positive_spans += len(gold_spans & predicted_spans)

        if len(previews) < preview_limit and gold_spans != predicted_spans:
            previews.append(
                {
                    "sample_id": sample.sample_id,
                    "review_text": sample.review_text,
                    "gold_aspects": spans_to_text(sample.tokens, sorted(gold_spans)),
                    "predicted_aspects": spans_to_text(sample.tokens, sorted(predicted_spans)),
                }
            )

    token_accuracy = (correct_tokens / total_tokens) if total_tokens else 0.0
    sequence_accuracy = (exact_sequence_matches / len(samples)) if samples else 0.0
    precision = (true_positive_spans / predicted_span_total) if predicted_span_total else 0.0
    recall = (true_positive_spans / gold_span_total) if gold_span_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "samples": len(samples),
        "tokens": total_tokens,
        "token_accuracy": round(token_accuracy, 4),
        "sequence_accuracy": round(sequence_accuracy, 4),
        "span_precision": round(precision, 4),
        "span_recall": round(recall, 4),
        "span_f1": round(f1, 4),
        "gold_span_total": gold_span_total,
        "predicted_span_total": predicted_span_total,
        "true_positive_spans": true_positive_spans,
        "preview_mismatches": previews,
    }


def main() -> None:
    args = parse_args()
    merge_reviews = not args.no_merge_reviews
    samples = load_bio_samples(args.input, merge_reviews=merge_reviews)
    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    model = HiddenMarkovAspectExtractor(
        smoothing=args.smoothing,
        min_token_freq=args.min_token_freq,
    )
    model.fit(train_samples)

    metrics = {
        "config": {
            "input": str(args.input),
            "seed": args.seed,
            "smoothing": args.smoothing,
            "min_token_freq": args.min_token_freq,
            "merge_reviews": merge_reviews,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        "dataset": {
            "total_samples": len(samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
        },
        "train": evaluate_split(model, train_samples, preview_limit=0),
        "validation": evaluate_split(model, val_samples, preview_limit=args.preview_limit),
        "test": evaluate_split(model, test_samples, preview_limit=args.preview_limit),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "hmm_aspect_extractor.json"
    metrics_path = args.output_dir / "metrics.json"

    model.save(model_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        f"Samples={len(samples)} train={len(train_samples)} val={len(val_samples)} "
        f"test={len(test_samples)}"
    )
    print(
        f"Validation span F1={metrics['validation']['span_f1']:.4f} "
        f"precision={metrics['validation']['span_precision']:.4f} "
        f"recall={metrics['validation']['span_recall']:.4f}"
    )
    print(
        f"Test span F1={metrics['test']['span_f1']:.4f} "
        f"precision={metrics['test']['span_precision']:.4f} "
        f"recall={metrics['test']['span_recall']:.4f}"
    )


if __name__ == "__main__":
    main()
