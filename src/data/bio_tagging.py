import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


TOKEN_PATTERN = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", flags=re.UNICODE)


ASPECT_KEYWORDS: Dict[str, Sequence[str]] = {
    "Accuracy": (
        "accuracy",
        "accurate",
        "inaccurate",
        "incorrect",
        "wrong",
        "hallucination",
        "hallucinations",
        "error",
        "errors",
        "false",
        "facts",
        "fact-checking",
        "fact checking",
        "reliable",
        "reliability",
        "random tokens",
        "calculations",
        "misleading",
        "truth",
        "valid",
    ),
    "Content": (
        "content",
        "response",
        "responses",
        "answer",
        "answers",
        "explanation",
        "explanations",
        "translation",
        "translations",
        "summary",
        "summaries",
        "writing",
        "document",
        "documents",
        "materials",
        "template",
        "templates",
        "interaction",
        "conversation",
        "wordy",
        "simple",
        "creative writing",
        "brainstorming",
    ),
    "Emotional Intelligence": (
        "friendly",
        "human",
        "empathy",
        "empathetic",
        "tone",
        "kind",
        "kindness",
        "frustrating",
        "understand",
        "understanding",
        "good faith",
        "good-faith",
        "conversation partner",
        "person",
        "conviction",
        "flip-flop",
        "healthier alternatives",
        "trusted adults",
        "frustrated",
    ),
    "Privacy": (
        "privacy",
        "private",
        "personal data",
        "data",
        "security",
        "secure",
        "tracking",
        "tracked",
        "scan",
        "police",
        "cloud-based",
        "cloud based",
        "trusting cloud",
        "track",
    ),
    "Subscription": (
        "subscription",
        "subscribed",
        "premium",
        "paid",
        "paying",
        "price",
        "pricing",
        "plan",
        "plans",
        "free",
        "unlimited",
        "value",
        "cost",
        "bucks",
    ),
    "Time": (
        "time",
        "slow",
        "slower",
        "fast",
        "faster",
        "quick",
        "quickly",
        "minute",
        "minutes",
        "hour",
        "hours",
        "latency",
        "long",
        "longer",
        "productivity",
        "faster",
        "takes",
        "within minutes",
        "back-and-forth",
    ),
    "UI (User Interface)": (
        "ui",
        "user interface",
        "interface",
        "layout",
        "design",
        "visual",
        "notifications",
        "feature",
        "features",
        "bug",
        "bugs",
        "glitch",
        "glitchy",
        "experience",
        "layout",
        "visual tweaks",
        "responsive",
        "harder to find",
    ),
    "Update": (
        "update",
        "updated",
        "updating",
        "recent update",
        "last update",
        "newest update",
        "version",
        "release",
        "released",
        "new version",
        "after updating",
    ),
    "Usefulness": (
        "useful",
        "usefulness",
        "helpful",
        "helps",
        "help",
        "benefit",
        "practical",
        "works",
        "working",
        "useless",
        "resources",
        "investigation",
        "daily habits",
        "to-do lists",
        "choose",
        "helps me",
        "not useful",
    ),
}


def tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens: List[str] = []
    offsets: List[Tuple[int, int]] = []
    for match in TOKEN_PATTERN.finditer(text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def _build_pattern(term: str) -> str:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return rf"(?<!\w){escaped}(?!\w)"


def find_keyword_spans(text: str, keywords: Iterable[str]) -> Tuple[List[Tuple[int, int]], List[str]]:
    spans: List[Tuple[int, int]] = []
    matched_terms: List[str] = []
    sorted_terms = sorted(set(keywords), key=len, reverse=True)
    for term in sorted_terms:
        pattern = _build_pattern(term)
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            spans.append((match.start(), match.end()))
            matched_terms.append(term)
    return merge_spans(spans), sorted(set(matched_terms))


def merge_spans(spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int]] = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def spans_to_bio(offsets: Sequence[Tuple[int, int]], spans: Sequence[Tuple[int, int]]) -> List[str]:
    tags = ["O"] * len(offsets)
    for span_start, span_end in spans:
        token_indices = []
        for idx, (tok_start, tok_end) in enumerate(offsets):
            overlap = tok_start < span_end and tok_end > span_start
            if overlap:
                token_indices.append(idx)
        if token_indices:
            tags[token_indices[0]] = "B-ASP"
            for idx in token_indices[1:]:
                tags[idx] = "I-ASP"
    return tags


def build_bio_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    output_rows = []
    label_counter: Counter = Counter()
    rows_with_tags = 0
    coverage_by_aspect: Counter = Counter()
    matched_term_counter: Counter = Counter()
    unmatched_by_aspect: Dict[str, List[Dict]] = {}

    for row_id, row in df.reset_index(drop=True).iterrows():
        review_text = str(row["review_text"])
        aspect = str(row["aspect"])
        keywords = ASPECT_KEYWORDS.get(aspect, ())
        spans, matched_terms = find_keyword_spans(review_text, keywords)
        tokens, offsets = tokenize_with_offsets(review_text)
        bio_tags = spans_to_bio(offsets, spans)

        has_aspect_tag = any(tag != "O" for tag in bio_tags)
        if has_aspect_tag:
            rows_with_tags += 1
            coverage_by_aspect[aspect] += 1
            matched_term_counter.update(matched_terms)
        else:
            examples = unmatched_by_aspect.setdefault(aspect, [])
            if len(examples) < 5:
                examples.append(
                    {
                        "row_id": row_id,
                        "argument": str(row["argument"]),
                        "sentiment": str(row["sentiment"]),
                    }
                )

        label_counter.update(bio_tags)
        output_rows.append(
            {
                "row_id": row_id,
                "ai_tools": row["ai_tools"],
                "aspect": aspect,
                "sentiment": row["sentiment"],
                "review_text": review_text,
                "argument": row["argument"],
                "tokens_json": json.dumps(tokens, ensure_ascii=False),
                "bio_tags_json": json.dumps(bio_tags, ensure_ascii=False),
                "matched_terms_json": json.dumps(matched_terms, ensure_ascii=False),
                "span_count": len(spans),
                "tag_status": "matched" if has_aspect_tag else "no_match",
            }
        )

    bio_df = pd.DataFrame(output_rows)
    total_rows = len(bio_df)
    coverage_pct = round((rows_with_tags / total_rows) * 100, 2) if total_rows else 0.0
    avg_tokens = round(
        bio_df["tokens_json"].map(lambda x: len(json.loads(x))).mean(), 2
    ) if total_rows else 0.0

    aspect_freq = df["aspect"].value_counts().sort_index().to_dict()
    sentiment_freq = df["sentiment"].value_counts().sort_index().to_dict()
    aspect_sentiment = (
        df.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0).sort_index().to_dict(orient="index")
    )

    report = {
        "total_rows": total_rows,
        "rows_with_aspect_tags": rows_with_tags,
        "rows_without_aspect_tags": total_rows - rows_with_tags,
        "coverage_pct": coverage_pct,
        "avg_tokens_per_row": avg_tokens,
        "bio_label_counts": dict(label_counter),
        "aspect_frequency": aspect_freq,
        "sentiment_distribution": sentiment_freq,
        "aspect_sentiment_distribution": aspect_sentiment,
        "coverage_by_aspect": dict(coverage_by_aspect),
        "coverage_by_aspect_pct": {
            aspect: round((coverage_by_aspect.get(aspect, 0) / count) * 100, 2)
            for aspect, count in aspect_freq.items()
        },
        "top_matched_terms": dict(matched_term_counter.most_common(30)),
        "unmatched_examples_by_aspect": unmatched_by_aspect,
    }
    return bio_df, report


def write_conll(bio_df: pd.DataFrame, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for _, row in bio_df.iterrows():
            tokens = json.loads(row["tokens_json"])
            tags = json.loads(row["bio_tags_json"])
            for tok, tag in zip(tokens, tags):
                f.write(f"{row['row_id']}\t{row['aspect']}\t{row['sentiment']}\t{tok}\t{tag}\n")
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create BIO tags for aspect terms in Mendeley ABSA dataset.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/mendeley.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/mendeley_bio_tagged.csv"))
    parser.add_argument("--output-conll", type=Path, default=Path("data/processed/mendeley_bio_tagged.conll"))
    parser.add_argument("--report-json", type=Path, default=Path("data/processed/mendeley_bio_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    bio_df, report = build_bio_dataset(df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_conll.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)

    bio_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    write_conll(bio_df, args.output_conll)
    with args.report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved BIO CSV: {args.output_csv}")
    print(f"Saved BIO CoNLL: {args.output_conll}")
    print(f"Saved report JSON: {args.report_json}")
    print(
        f"Coverage: {report['rows_with_aspect_tags']}/{report['total_rows']} "
        f"({report['coverage_pct']}%) rows contain B/I aspect tags."
    )


if __name__ == "__main__":
    main()
