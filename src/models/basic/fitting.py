import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .aspect_detector import AspectDetector
from .constants import FINAL_CATEGORIES, INTENSIFIERS, NEGATIONS, STOP_WORDS
from .sentiment_rules import SentimentAssigner
from .tokenizer import technical_tokenize


SUPPORTED_SENTIMENTS = frozenset({"Positive", "Negative", "Neutral"})
ADAPTATION_KEYS = frozenset(
    {"aspect_terms", "category_terms", "relevance_terms", "sentiment_terms"}
)


@dataclass(frozen=True)
class GoldQuad:
    aspect: str
    category: str
    opinion: str
    sentiment: str

    def as_dict(self) -> dict[str, str]:
        return {
            "aspect": self.aspect,
            "category": self.category,
            "opinion": self.opinion,
            "sentiment": self.sentiment,
        }


@dataclass(frozen=True)
class GoldExample:
    sentence: str
    quads: tuple[GoldQuad, ...]


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def load_gold_jsonl(path: str | Path) -> list[GoldExample]:
    examples: list[GoldExample] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            sentence = str(raw.get("sentence", "")).strip()
            if not sentence:
                continue
            quads: list[GoldQuad] = []
            seen: set[tuple[str, str, str, str]] = set()
            for item in raw.get("quads", []):
                category = str(item.get("category", "")).strip().upper()
                sentiment = str(item.get("sentiment", "")).strip().title()
                quad = GoldQuad(
                    aspect=str(item.get("aspect", "")).strip(),
                    category=category,
                    opinion=str(item.get("opinion", "")).strip(),
                    sentiment=sentiment,
                )
                key = (
                    normalize(quad.aspect),
                    quad.category,
                    normalize(quad.opinion),
                    quad.sentiment,
                )
                if (
                    not quad.aspect
                    or category not in FINAL_CATEGORIES
                    or sentiment not in SUPPORTED_SENTIMENTS
                    or key in seen
                ):
                    continue
                seen.add(key)
                quads.append(quad)
            if quads:
                examples.append(GoldExample(sentence, tuple(quads)))
    return examples


def deduplicate_examples(examples: Iterable[GoldExample]) -> list[GoldExample]:
    grouped: dict[str, dict[tuple[str, str, str, str], GoldQuad]] = {}
    sentence_text: dict[str, str] = {}
    for example in examples:
        sentence_key = normalize(example.sentence)
        sentence_text.setdefault(sentence_key, example.sentence)
        grouped.setdefault(sentence_key, {})
        for quad in example.quads:
            key = (
                normalize(quad.aspect), quad.category,
                normalize(quad.opinion), quad.sentiment,
            )
            grouped[sentence_key][key] = quad
    return [
        GoldExample(sentence_text[key], tuple(grouped[key].values()))
        for key in grouped
    ]


class BaselineFitter:
    """Corpus adaptation for the rule baseline; no gradient optimization."""

    def __init__(
        self,
        min_term_frequency: int = 3,
        category_purity: float = 0.65,
        max_link_distance: int = 12,
        sentiment_threshold: float = 0.15,
    ):
        self.min_term_frequency = min_term_frequency
        self.category_purity = category_purity
        self.max_link_distance = max_link_distance
        self.sentiment_threshold = sentiment_threshold
        self.aspect_detector = AspectDetector()
        self.sentiment_assigner = SentimentAssigner()

    def fit(
        self,
        train_examples: Iterable[GoldExample],
        validation_examples: Iterable[GoldExample] = (),
        reviewed_adaptations: dict | None = None,
    ) -> tuple[dict, dict, list[GoldExample]]:
        raw_train = deduplicate_examples(train_examples)
        validation = deduplicate_examples(validation_examples)
        validation_sentences = {normalize(item.sentence) for item in validation}
        train = [
            item for item in raw_train
            if normalize(item.sentence) not in validation_sentences
        ]

        adaptations = reviewed_adaptations or {}
        aspect_candidates = self._aspect_candidates(train)
        category_candidates = self._category_candidates(train)
        relevance_candidates = self._relevance_candidates(train)
        so_pmi_candidates = self._so_pmi_candidates(train)
        supervised_opinions = self._supervised_opinion_candidates(train)

        learned_aspect_terms = {
            normalize(term): str(category).strip().upper()
            for term, category in adaptations.get("aspect_terms", {}).items()
            if normalize(term) and str(category).strip().upper() in FINAL_CATEGORIES
        }
        learned_category_terms = {
            normalize(term): str(category).strip().upper()
            for term, category in adaptations.get("category_terms", {}).items()
            if normalize(term) and str(category).strip().upper() in FINAL_CATEGORIES
        }
        learned_relevance_terms = sorted(
            {
                normalize(term)
                for term in adaptations.get("relevance_terms", [])
                if normalize(str(term))
            }
        )
        promoted = {
            normalize(term): float(score)
            for term, score in adaptations.get("sentiment_terms", {}).items()
            if normalize(term)
        }
        created_at = datetime.now(timezone.utc).isoformat()

        artifact = {
            "schema_version": 1,
            "model_name": "ai-absa-rule-baseline-v1",
            "created_at": created_at,
            "thresholds": {
                "max_link_distance": self.max_link_distance,
                "sentiment": self.sentiment_threshold,
            },
            "learned_category_terms": learned_category_terms,
            "learned_aspect_terms": learned_aspect_terms,
            "learned_relevance_terms": learned_relevance_terms,
            "promoted_sentiment_terms": promoted,
            "metadata": {
                "schema_version": 1,
                "training_type": "controlled_rule_fitting",
                "gradient_training": False,
                "train_sentences_before_overlap_removal": len(raw_train),
                "train_sentences": len(train),
                "train_quads": sum(len(item.quads) for item in train),
                "validation_sentences": len(validation),
                "train_validation_overlap_removed": len(raw_train) - len(train),
                "category_terms_activated_after_review": len(learned_category_terms),
                "aspect_terms_activated_after_review": len(learned_aspect_terms),
                "relevance_terms_activated_after_review": len(learned_relevance_terms),
                "sentiment_terms_activated_after_review": len(promoted),
                "categories": list(FINAL_CATEGORIES),
                "candidate_counts": {
                    "aspect": len(aspect_candidates),
                    "category": len(category_candidates),
                    "relevance": len(relevance_candidates),
                    "so_pmi": len(so_pmi_candidates),
                    "supervised_opinion": len(supervised_opinions),
                },
            },
        }
        report = {
            "created_at": created_at,
            "policy": (
                "Aspect, category, relevance, and SO-PMI terms are candidates "
                "only. Production lexicons change only through an explicit "
                "reviewed-adaptations file."
            ),
            "aspect_candidates": aspect_candidates[:500],
            "category_candidates": category_candidates[:500],
            "relevance_candidates": relevance_candidates[:500],
            "so_pmi_candidates": so_pmi_candidates[:500],
            "supervised_opinion_candidates": supervised_opinions[:500],
        }
        return artifact, report, train

    def _aspect_candidates(self, examples: list[GoldExample]) -> list[dict]:
        excluded = {
            "it", "this", "they", "the model", "this model", "model", "models",
            "the ones", "one", "that", "he", "she",
        }
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for example in examples:
            for quad in example.quads:
                aspect = normalize(quad.aspect)
                if (
                    aspect in excluded
                    or len(aspect) < 3
                    or len(aspect.split()) > 6
                ):
                    continue
                counts[aspect][quad.category] += 1
        rows: list[dict] = []
        for aspect, category_counts in counts.items():
            frequency = sum(category_counts.values())
            category, support = category_counts.most_common(1)[0]
            rows.append(
                {
                    "term": aspect,
                    "category": category,
                    "frequency": frequency,
                    "purity": support / frequency,
                    "distribution": dict(category_counts),
                }
            )
        return sorted(
            rows,
            key=lambda row: (-row["frequency"], -row["purity"], row["term"]),
        )

    def _category_candidates(self, examples: list[GoldExample]) -> list[dict]:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for example in examples:
            for quad in example.quads:
                terms = self._candidate_terms(f"{quad.aspect} {quad.opinion}")
                phrases = {normalize(quad.aspect), normalize(quad.opinion)}
                for term in terms | {phrase for phrase in phrases if phrase}:
                    counts[term][quad.category] += 1

        rows: list[dict] = []
        for term, category_counts in counts.items():
            frequency = sum(category_counts.values())
            category, support = category_counts.most_common(1)[0]
            rows.append(
                {
                    "term": term,
                    "category": category,
                    "frequency": frequency,
                    "purity": support / frequency,
                    "distribution": dict(category_counts),
                }
            )
        return sorted(
            rows,
            key=lambda row: (-row["frequency"], -row["purity"], row["term"]),
        )

    def _so_pmi_candidates(self, examples: list[GoldExample]) -> list[dict]:
        term_df: Counter[str] = Counter()
        positive_df: Counter[str] = Counter()
        negative_df: Counter[str] = Counter()
        positive_events = 0
        negative_events = 0

        for example in examples:
            clean_text, tokens = technical_tokenize(example.sentence)
            terms = self._candidate_terms(clean_text)
            signals = self.sentiment_assigner.detect_signals(clean_text, tokens)
            has_positive = any(signal.score > 0 for signal in signals)
            has_negative = any(signal.score < 0 for signal in signals)
            positive_events += int(has_positive)
            negative_events += int(has_negative)
            term_df.update(terms)
            if has_positive:
                positive_df.update(terms)
            if has_negative:
                negative_df.update(terms)

        total = len(examples)
        alpha = 1.0
        rows: list[dict] = []
        for term, frequency in term_df.items():
            if frequency < self.min_term_frequency:
                continue
            pmi_positive = math.log(
                ((positive_df[term] + alpha) * (total + alpha))
                / ((frequency + alpha) * (positive_events + alpha))
            )
            pmi_negative = math.log(
                ((negative_df[term] + alpha) * (total + alpha))
                / ((frequency + alpha) * (negative_events + alpha))
            )
            orientation = pmi_positive - pmi_negative
            rows.append(
                {
                    "term": term,
                    "frequency": frequency,
                    "positive_cooccurrence": positive_df[term],
                    "negative_cooccurrence": negative_df[term],
                    "so_pmi": orientation,
                    "suggested_sentiment": (
                        "Positive" if orientation > 0 else "Negative"
                    ),
                }
            )
        return sorted(
            rows,
            key=lambda row: (-abs(row["so_pmi"]), -row["frequency"], row["term"]),
        )

    def _relevance_candidates(self, examples: list[GoldExample]) -> list[dict]:
        counts: Counter[str] = Counter()
        for example in examples:
            counts.update(self._candidate_terms(example.sentence))
        rows = [
            {"term": term, "frequency": frequency}
            for term, frequency in counts.items()
            if frequency >= self.min_term_frequency
        ]
        return sorted(rows, key=lambda row: (-row["frequency"], row["term"]))

    @staticmethod
    def _supervised_opinion_candidates(examples: list[GoldExample]) -> list[dict]:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for example in examples:
            for quad in example.quads:
                opinion = normalize(quad.opinion)
                if opinion:
                    counts[opinion][quad.sentiment] += 1
        rows: list[dict] = []
        for opinion, sentiment_counts in counts.items():
            frequency = sum(sentiment_counts.values())
            sentiment, support = sentiment_counts.most_common(1)[0]
            rows.append(
                {
                    "opinion": opinion,
                    "sentiment": sentiment,
                    "frequency": frequency,
                    "purity": support / frequency,
                    "distribution": dict(sentiment_counts),
                }
            )
        return sorted(rows, key=lambda row: (-row["frequency"], row["opinion"]))

    def _candidate_terms(self, text: str) -> set[str]:
        clean_text, tokens = technical_tokenize(text)
        model_spans = [
            (candidate.start, candidate.end)
            for candidate in self.aspect_detector.detect(clean_text, tokens)
            if candidate.group == "model"
        ]
        terms: set[str] = set()
        for token in tokens:
            term = token.normalized
            if (
                len(term) < 3
                or term in STOP_WORDS
                or term in NEGATIONS
                or term in INTENSIFIERS
                or term.isdigit()
                or not re.search(r"[a-z]", term)
                or any(start <= token.start < end for start, end in model_spans)
            ):
                continue
            terms.add(term)
        return terms


def load_reviewed_adaptations(path: str | Path | None) -> dict:
    if path is None:
        return {
            "aspect_terms": {},
            "category_terms": {},
            "relevance_terms": [],
            "sentiment_terms": {},
        }
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Reviewed adaptations must be a JSON object")

    # Backward compatibility with the old flat sentiment term -> score file.
    if not (set(raw) & ADAPTATION_KEYS):
        raw = {"sentiment_terms": raw}

    unknown = set(raw) - ADAPTATION_KEYS
    if unknown:
        raise ValueError(f"Unknown reviewed adaptation keys: {sorted(unknown)}")

    aspect_terms = raw.get("aspect_terms", {})
    category_terms = raw.get("category_terms", {})
    relevance_terms = raw.get("relevance_terms", [])
    sentiment_terms = raw.get("sentiment_terms", {})
    if not isinstance(aspect_terms, dict) or not isinstance(category_terms, dict):
        raise ValueError("aspect_terms and category_terms must be JSON objects")
    if not isinstance(relevance_terms, list):
        raise ValueError("relevance_terms must be a JSON array")
    if not isinstance(sentiment_terms, dict):
        raise ValueError("sentiment_terms must be a JSON object")
    return {
        "aspect_terms": {str(k): str(v) for k, v in aspect_terms.items()},
        "category_terms": {str(k): str(v) for k, v in category_terms.items()},
        "relevance_terms": [str(term) for term in relevance_terms],
        "sentiment_terms": {
            str(term): float(score) for term, score in sentiment_terms.items()
        },
    }


def load_reviewed_promotions(path: str | Path | None) -> dict[str, float]:
    """Compatibility helper for the previous sentiment-only review format."""
    return load_reviewed_adaptations(path)["sentiment_terms"]
