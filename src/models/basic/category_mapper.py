from collections import Counter

from .baseline_types import AspectCandidate, SentimentSignal
from .constants import CATEGORY_KEYWORDS, FINAL_CATEGORIES


_GROUP_CATEGORY = {
    "behavior": "BEHAVIOR",
    "intelligence": "INTELLIGENCE",
    "performance": "PERFORMANCE",
    "resources": "RESOURCES",
    "software": "SOFTWARE",
    "technical": "TECHNICAL",
}


class CategoryMapper:
    def __init__(self, learned_terms: dict[str, str] | None = None):
        self.learned_terms = learned_terms or {}

    def map(
        self,
        aspect: AspectCandidate,
        evidence: tuple[SentimentSignal, ...],
        clause_text: str,
    ) -> str:
        combined = " ".join(
            [aspect.text, clause_text, *(signal.text for signal in evidence)]
        ).casefold()
        scores: Counter[str] = Counter()

        for term, category in self.learned_terms.items():
            if term in combined and category in FINAL_CATEGORIES:
                scores[category] += 2
        for category, keywords in CATEGORY_KEYWORDS.items():
            scores[category] += sum(keyword in combined for keyword in keywords)

        group_category = _GROUP_CATEGORY.get(aspect.group)
        if group_category:
            scores[group_category] += 2
        if scores:
            return max(FINAL_CATEGORIES, key=lambda category: (scores[category], -FINAL_CATEGORIES.index(category)))
        return "TECHNICAL"
