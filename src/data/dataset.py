import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

import pandas as pd


BIO_TAGS = ("B-ASP", "I-ASP", "O")
ASPECT_TAG_SET = {"B-ASP", "I-ASP"}


@dataclass(frozen=True)
class BioSequenceSample:
    sample_id: str
    review_text: str
    tokens: List[str]
    bio_tags: List[str]
    source_row_ids: List[int]
    aspects: List[str]
    sentiments: List[str]
    matched_terms: List[str]


class AbsaDataset:
    def __init__(self, samples: Sequence[BioSequenceSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BioSequenceSample:
        return self.samples[index]

    @classmethod
    def from_bio_csv(cls, csv_path: str | Path, merge_reviews: bool = True) -> "AbsaDataset":
        return cls(load_bio_samples(csv_path, merge_reviews=merge_reviews))


def _parse_json_list(value: str) -> List[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got: {type(parsed)!r}")
    return [str(item) for item in parsed]


def bio_tags_to_token_spans(tags: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = None

    for idx, tag in enumerate(tags):
        if tag == "B-ASP":
            if start is not None:
                spans.append((start, idx))
            start = idx
        elif tag == "I-ASP":
            if start is None:
                start = idx
        else:
            if start is not None:
                spans.append((start, idx))
                start = None

    if start is not None:
        spans.append((start, len(tags)))
    return spans


def merge_overlapping_token_spans(spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda item: (item[0], item[1]))
    merged: List[Tuple[int, int]] = [sorted_spans[0]]

    for start, end in sorted_spans[1:]:
        last_start, last_end = merged[-1]
        if start < last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def token_spans_to_bio(length: int, spans: Sequence[Tuple[int, int]]) -> List[str]:
    tags = ["O"] * length
    for start, end in spans:
        if not 0 <= start < end <= length:
            raise ValueError(f"Invalid token span: {(start, end)} for length={length}")
        tags[start] = "B-ASP"
        for idx in range(start + 1, end):
            tags[idx] = "I-ASP"
    return tags


def _row_to_sample(row: Mapping[str, object]) -> BioSequenceSample:
    tokens = _parse_json_list(str(row["tokens_json"]))
    bio_tags = _parse_json_list(str(row["bio_tags_json"]))
    matched_terms = _parse_json_list(str(row["matched_terms_json"]))
    if len(tokens) != len(bio_tags):
        raise ValueError(f"Token/tag length mismatch for row_id={row['row_id']}")

    return BioSequenceSample(
        sample_id=f"row-{int(row['row_id'])}",
        review_text=str(row["review_text"]),
        tokens=tokens,
        bio_tags=bio_tags,
        source_row_ids=[int(row["row_id"])],
        aspects=[str(row["aspect"])],
        sentiments=[str(row["sentiment"])],
        matched_terms=sorted(set(matched_terms)),
    )


def _merge_review_rows(group_id: int, rows: pd.DataFrame) -> BioSequenceSample:
    first = rows.iloc[0]
    tokens = _parse_json_list(str(first["tokens_json"]))
    token_spans: List[Tuple[int, int]] = []
    aspects = set()
    sentiments = set()
    matched_terms = set()
    source_row_ids: List[int] = []

    for _, row in rows.iterrows():
        row_tokens = _parse_json_list(str(row["tokens_json"]))
        row_tags = _parse_json_list(str(row["bio_tags_json"]))
        row_terms = _parse_json_list(str(row["matched_terms_json"]))

        if row_tokens != tokens:
            raise ValueError("Found inconsistent tokenization inside the same review_text group.")
        if len(row_tags) != len(tokens):
            raise ValueError(f"Token/tag length mismatch for row_id={row['row_id']}")

        token_spans.extend(bio_tags_to_token_spans(row_tags))
        aspects.add(str(row["aspect"]))
        sentiments.add(str(row["sentiment"]))
        matched_terms.update(row_terms)
        source_row_ids.append(int(row["row_id"]))

    merged_spans = merge_overlapping_token_spans(token_spans)
    merged_tags = token_spans_to_bio(len(tokens), merged_spans)

    return BioSequenceSample(
        sample_id=f"review-{group_id}",
        review_text=str(first["review_text"]),
        tokens=tokens,
        bio_tags=merged_tags,
        source_row_ids=sorted(source_row_ids),
        aspects=sorted(aspects),
        sentiments=sorted(sentiments),
        matched_terms=sorted(matched_terms),
    )


def load_bio_samples(csv_path: str | Path, merge_reviews: bool = True) -> List[BioSequenceSample]:
    df = pd.read_csv(csv_path)
    if merge_reviews:
        return [
            _merge_review_rows(group_id, rows)
            for group_id, (_, rows) in enumerate(df.groupby("review_text", sort=False))
        ]
    return [_row_to_sample(row) for row in df.to_dict(orient="records")]


def split_samples(
    samples: Sequence[BioSequenceSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[BioSequenceSample], List[BioSequenceSample], List[BioSequenceSample]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_samples = shuffled[:train_end]
    val_samples = shuffled[train_end:val_end]
    test_samples = shuffled[val_end:]
    return train_samples, val_samples, test_samples


def create_target_string(triplets: Iterable[Mapping[str, object]]) -> List[str]:
    result: List[str] = []
    sorted_triplets = sorted(triplets, key=lambda item: item.get("start_idx", -1))
    for triplet in sorted_triplets:
        result.append(
            f"[ASP] {triplet.get('aspect', '')} [OP] {triplet.get('opinion', '')} "
            f"[SENT] {triplet.get('sentiment', '')} [EOS]"
        )
    return result
