"""Dataset statistics for the AI-ABSA dashboard."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


CATEGORY_OPTIONS = [
    "PERFORMANCE",
    "INTELLIGENCE",
    "RESOURCES",
    "BEHAVIOR",
    "TECHNICAL",
    "SOFTWARE",
    "COMPARATIVE",
]

SENTIMENT_OPTIONS = [
    "Positive",
    "Negative",
    "Neutral",
]


def _iter_quads(rows: list[dict[str, Any]]):
    for row in rows:
        for quad in row.get("quads", []):
            yield quad


def _percent(count: int, total: int) -> float:
    return round((count / total * 100) if total else 0, 2)


def compute_overview(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_sentences = len(rows)
    quad_counts = [len(row.get("quads", [])) for row in rows]
    total_quads = sum(quad_counts)

    return {
        "total_sentences": total_sentences,
        "total_quads": total_quads,
        "sentences_with_quads": sum(1 for count in quad_counts if count > 0),
        "sentences_without_quads": sum(1 for count in quad_counts if count == 0),
        "single_quad_sentences": sum(1 for count in quad_counts if count == 1),
        "multi_quad_sentences": sum(1 for count in quad_counts if count > 1),
        "avg_quads_per_sentence": round(total_quads / total_sentences, 3)
        if total_sentences
        else 0,
        "categories": len(CATEGORY_OPTIONS),
        "sentiments": len(SENTIMENT_OPTIONS),
    }


def compute_category_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {category: 0 for category in CATEGORY_OPTIONS}
    unknown_count = 0
    for quad in _iter_quads(rows):
        category = quad.get("category", "")
        if category in counts:
            counts[category] += 1
        else:
            unknown_count += 1

    total_known = sum(counts.values())
    return [
        {
            "category": category,
            "count": counts[category],
            "percent": _percent(counts[category], total_known),
        }
        for category in CATEGORY_OPTIONS
    ]


def compute_sentiment_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {sentiment: 0 for sentiment in SENTIMENT_OPTIONS}
    unknown_count = 0
    for quad in _iter_quads(rows):
        sentiment = quad.get("sentiment", "")
        if sentiment in counts:
            counts[sentiment] += 1
        else:
            unknown_count += 1

    total_known = sum(counts.values())
    return [
        {
            "sentiment": sentiment,
            "count": counts[sentiment],
            "percent": _percent(counts[sentiment], total_known),
        }
        for sentiment in SENTIMENT_OPTIONS
    ]


def compute_top_aspects(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    aspect_stats: dict[str, dict[str, Any]] = {}

    for quad in _iter_quads(rows):
        aspect = str(quad.get("aspect") or "").strip()
        if not aspect:
            continue

        key = aspect.lower()
        if key not in aspect_stats:
            aspect_stats[key] = {
                "aspect": aspect,
                "count": 0,
                "categories": {category: 0 for category in CATEGORY_OPTIONS},
                "sentiments": {sentiment: 0 for sentiment in SENTIMENT_OPTIONS},
            }

        stats = aspect_stats[key]
        stats["count"] += 1

        category = quad.get("category", "")
        if category in stats["categories"]:
            stats["categories"][category] += 1

        sentiment = quad.get("sentiment", "")
        if sentiment in stats["sentiments"]:
            stats["sentiments"][sentiment] += 1

    ranked = sorted(aspect_stats.values(), key=lambda item: item["count"], reverse=True)
    results: list[dict[str, Any]] = []

    for item in ranked[: max(1, min(int(limit or 12), 50))]:
        top_category = max(item["categories"].items(), key=lambda pair: pair[1])[0]
        top_sentiment = max(item["sentiments"].items(), key=lambda pair: pair[1])[0]
        results.append(
            {
                "aspect": item["aspect"],
                "count": item["count"],
                "top_category": top_category,
                "top_sentiment": top_sentiment,
                "categories": item["categories"],
                "sentiments": item["sentiments"],
            }
        )

    return results


def compute_category_sentiment_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix = {
        category: {sentiment: 0 for sentiment in SENTIMENT_OPTIONS}
        for category in CATEGORY_OPTIONS
    }

    for quad in _iter_quads(rows):
        category = quad.get("category", "")
        sentiment = quad.get("sentiment", "")
        if category in matrix and sentiment in SENTIMENT_OPTIONS:
            matrix[category][sentiment] += 1

    return [
        {
            "category": category,
            **matrix[category],
            "total": sum(matrix[category].values()),
        }
        for category in CATEGORY_OPTIONS
    ]


def _rank_counter(
    counter: Counter,
    labels: dict[str, str] | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    total = sum(counter.values())
    label_map = labels or {}
    ranked = sorted(counter.items(), key=lambda pair: (-pair[1], label_map.get(pair[0], pair[0]).lower()))
    return [
        {
            "key": key,
            "item": label_map.get(key, key),
            "count": count,
            "percent": _percent(count, total),
        }
        for key, count in ranked[: max(1, min(int(limit or 8), 20))]
        if count > 0
    ]


def compute_factor_focus(
    rows: list[dict[str, Any]],
    main_limit: int = 30,
    related_limit: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Build the cross-factor view used by the benchmark focus explorer."""
    category_sentiments = {category: Counter() for category in CATEGORY_OPTIONS}
    category_aspects = {category: Counter() for category in CATEGORY_OPTIONS}
    sentiment_categories = {sentiment: Counter() for sentiment in SENTIMENT_OPTIONS}
    sentiment_aspects = {sentiment: Counter() for sentiment in SENTIMENT_OPTIONS}
    aspect_stats: dict[str, dict[str, Any]] = {}

    for quad in _iter_quads(rows):
        category = quad.get("category", "")
        sentiment = quad.get("sentiment", "")
        aspect = str(quad.get("aspect") or "").strip()
        aspect_key = aspect.lower()

        if aspect and aspect_key not in aspect_stats:
            aspect_stats[aspect_key] = {
                "aspect": aspect,
                "count": 0,
                "categories": Counter(),
                "sentiments": Counter(),
            }

        if category in CATEGORY_OPTIONS and sentiment in SENTIMENT_OPTIONS:
            category_sentiments[category][sentiment] += 1
            sentiment_categories[sentiment][category] += 1

        if aspect:
            stats = aspect_stats[aspect_key]
            stats["count"] += 1
            if category in CATEGORY_OPTIONS:
                category_aspects[category][aspect_key] += 1
                stats["categories"][category] += 1
            if sentiment in SENTIMENT_OPTIONS:
                sentiment_aspects[sentiment][aspect_key] += 1
                stats["sentiments"][sentiment] += 1

    aspect_labels = {key: item["aspect"] for key, item in aspect_stats.items()}

    category_rows = []
    for category in CATEGORY_OPTIONS:
        sentiments = category_sentiments[category]
        support = sum(sentiments.values())
        category_rows.append(
            {
                "key": category,
                "item": category,
                "support": support,
                "Positive": sentiments["Positive"],
                "Negative": sentiments["Negative"],
                "Neutral": sentiments["Neutral"],
                "sentiments": _rank_counter(sentiments, limit=related_limit),
                "aspects": _rank_counter(category_aspects[category], aspect_labels, related_limit),
            }
        )

    sentiment_rows = []
    for sentiment in SENTIMENT_OPTIONS:
        categories = sentiment_categories[sentiment]
        support = sum(categories.values())
        sentiment_rows.append(
            {
                "key": sentiment,
                "item": sentiment,
                "support": support,
                "Positive": support if sentiment == "Positive" else 0,
                "Negative": support if sentiment == "Negative" else 0,
                "Neutral": support if sentiment == "Neutral" else 0,
                "categories": _rank_counter(categories, limit=related_limit),
                "aspects": _rank_counter(sentiment_aspects[sentiment], aspect_labels, related_limit),
            }
        )

    ranked_aspects = sorted(aspect_stats.items(), key=lambda pair: (-pair[1]["count"], pair[1]["aspect"].lower()))
    safe_main_limit = max(1, min(int(main_limit or 30), 100))
    aspect_rows = []
    for key, item in ranked_aspects[:safe_main_limit]:
        sentiments = item["sentiments"]
        aspect_rows.append(
            {
                "key": key,
                "item": item["aspect"],
                "support": item["count"],
                "Positive": sentiments["Positive"],
                "Negative": sentiments["Negative"],
                "Neutral": sentiments["Neutral"],
                "categories": _rank_counter(item["categories"], limit=related_limit),
                "sentiments": _rank_counter(sentiments, limit=related_limit),
            }
        )

    return {
        "category": category_rows,
        "sentiment": sentiment_rows,
        "aspect": aspect_rows,
    }


def compute_backend_unknowns(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count out-of-contract values without surfacing them in primary charts."""
    unknown_categories = 0
    unknown_sentiments = 0
    for quad in _iter_quads(rows):
        if quad.get("category", "") not in CATEGORY_OPTIONS:
            unknown_categories += 1
        if quad.get("sentiment", "") not in SENTIMENT_OPTIONS:
            unknown_sentiments += 1
    return {
        "UNKNOWN_category": unknown_categories,
        "UNKNOWN_sentiment": unknown_sentiments,
    }


def filter_samples(
    rows: list[dict[str, Any]],
    search: str | None = None,
    category: str | None = None,
    sentiment: str | None = None,
    quad_type: str = "all",
) -> list[dict[str, Any]]:
    search_text = (search or "").strip().lower()
    category = (category or "").strip()
    sentiment = (sentiment or "").strip()
    quad_type = quad_type if quad_type in {"all", "no_quad", "single_quad", "multi_quad"} else "all"

    def matches(row: dict[str, Any]) -> bool:
        quads = row.get("quads", [])
        quad_count = len(quads)

        if quad_type == "no_quad" and quad_count != 0:
            return False
        if quad_type == "single_quad" and quad_count != 1:
            return False
        if quad_type == "multi_quad" and quad_count <= 1:
            return False

        if search_text:
            haystack_parts = [row.get("sentence", "")]
            for quad in quads:
                haystack_parts.extend([quad.get("aspect", ""), quad.get("opinion", "")])
            if search_text not in " ".join(haystack_parts).lower():
                return False

        if category and not any(quad.get("category") == category for quad in quads):
            return False

        if sentiment and not any(quad.get("sentiment") == sentiment for quad in quads):
            return False

        return True

    return [row for row in rows if matches(row)]


def paginate(items: list[dict[str, Any]], page: int = 1, page_size: int = 20) -> dict[str, Any]:
    safe_page_size = max(1, min(int(page_size or 20), 100))
    total = len(items)
    total_pages = max(1, math.ceil(total / safe_page_size))
    safe_page = max(1, min(int(page or 1), total_pages))
    start = (safe_page - 1) * safe_page_size
    end = start + safe_page_size

    return {
        "page": safe_page,
        "page_size": safe_page_size,
        "total": total,
        "total_pages": total_pages,
        "items": items[start:end],
    }
