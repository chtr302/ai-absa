import argparse
import json
from pathlib import Path

from src.evaluation.rule_baseline import evaluate_rule_baseline
from src.models.basic.fitting import (
    BaselineFitter,
    load_gold_jsonl,
    load_reviewed_adaptations,
)
from src.models.basic.sentiment_model import ABSASentimentModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the report-aligned rule-based ABSA baseline. This performs "
            "controlled corpus adaptation, not gradient/GPU training."
        )
    )
    parser.add_argument(
        "--train-path",
        default="data/processed/final_data/train_final.jsonl",
    )
    parser.add_argument(
        "--val-path",
        default="data/processed/final_data/val_final.jsonl",
    )
    parser.add_argument("--output-dir", default="models/baseline")
    parser.add_argument(
        "--reviewed-adaptations",
        help=(
            "JSON file containing reviewed aspect/category/relevance/sentiment "
            "adaptations. Candidates are never activated without this file."
        ),
    )
    parser.add_argument(
        "--reviewed-promotions",
        help="Deprecated sentiment-only review file; kept for compatibility.",
    )
    parser.add_argument("--min-term-frequency", type=int, default=3)
    parser.add_argument("--category-purity", type=float, default=0.65)
    parser.add_argument("--max-link-distance", type=int, default=12)
    parser.add_argument("--sentiment-threshold", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = load_gold_jsonl(args.train_path)
    validation_examples = load_gold_jsonl(args.val_path)
    fitter = BaselineFitter(
        min_term_frequency=args.min_term_frequency,
        category_purity=args.category_purity,
        max_link_distance=args.max_link_distance,
        sentiment_threshold=args.sentiment_threshold,
    )
    review_path = args.reviewed_adaptations or args.reviewed_promotions
    artifact, fitting_report, cleaned_train = fitter.fit(
        train_examples,
        validation_examples,
        reviewed_adaptations=load_reviewed_adaptations(review_path),
    )

    model_path = output_dir / "model.json"
    report_path = output_dir / "fitting_report.json"
    metrics_path = output_dir / "validation_metrics.json"
    predictions_path = output_dir / "validation_predictions.jsonl"

    model_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(fitting_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model = ABSASentimentModel.from_artifact(model_path)
    metrics, predictions = evaluate_rule_baseline(model, validation_examples)
    metrics["data"] = {
        "train_sentences": len(cleaned_train),
        "train_quads": sum(len(item.quads) for item in cleaned_train),
        "validation_sentences": len(validation_examples),
        "validation_quads": sum(len(item.quads) for item in validation_examples),
        "train_validation_overlap_removed": artifact["metadata"][
            "train_validation_overlap_removed"
        ],
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with predictions_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "training_type": "controlled_rule_fitting",
                "model": str(model_path),
                "fitting_report": str(report_path),
                "validation_metrics": str(metrics_path),
                "validation_predictions": str(predictions_path),
                "summary": {
                    "aspect_f1": metrics["aspect"]["f1"],
                    "aspect_category_f1": metrics["aspect_category"]["f1"],
                    "aspect_sentiment_f1": metrics["aspect_sentiment"]["f1"],
                    "full_quad_f1": metrics["full_quad"]["f1"],
                    "sentiment_macro_f1_on_matched_aspects": metrics[
                        "sentiment_on_matched_aspects"
                    ]["macro_f1"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
