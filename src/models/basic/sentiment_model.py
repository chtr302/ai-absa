import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .aspect_detector import AspectDetector
from .baseline_types import AspectCandidate, LinkedAspect, SentimentSignal, TechnicalToken
from .category_mapper import CategoryMapper
from .relevance import is_domain_relevant
from .sentiment_rules import SentimentAssigner
from .tokenizer import clause_index, segment_clauses, technical_tokenize


class ABSASentimentModel:
    """Deterministic multi-quad ABSA baseline described in the project report."""

    MODEL_NAME = "ai-absa-rule-baseline-v1"
    POSITIVE = SentimentAssigner.POSITIVE
    NEGATIVE = SentimentAssigner.NEGATIVE
    NEUTRAL = SentimentAssigner.NEUTRAL

    def __init__(
        self,
        max_link_distance: int = 12,
        sentiment_threshold: float = 0.15,
        learned_category_terms: dict[str, str] | None = None,
        learned_aspect_terms: dict[str, str] | None = None,
        learned_relevance_terms: list[str] | None = None,
        promoted_sentiment_terms: dict[str, float] | None = None,
        artifact_metadata: dict[str, Any] | None = None,
    ):
        self.max_link_distance = max_link_distance
        self.sentiment_threshold = sentiment_threshold
        self.aspect_detector = AspectDetector(learned_aspects=learned_aspect_terms)
        promoted_patterns = tuple(
            (rf"\b{__import__('re').escape(term)}\b", score)
            for term, score in (promoted_sentiment_terms or {}).items()
        )
        self.sentiment_assigner = SentimentAssigner(
            promoted_patterns=promoted_patterns
        )
        self.category_mapper = CategoryMapper(learned_category_terms)
        self.learned_relevance_terms = tuple(
            term.casefold() for term in (learned_relevance_terms or []) if term
        )
        self.artifact_metadata = artifact_metadata or {}

    def predict(
        self,
        text: str,
        parent_context: str = "",
        thread_title: str = "",
        include_debug: bool = False,
    ) -> dict[str, Any]:
        clean_text, tokens = technical_tokenize(text)
        if not clean_text:
            return self._output(clean_text, [])

        sentence_aspects = self.aspect_detector.detect(clean_text, tokens)
        context_aspects = self._context_aspects(parent_context, thread_title)
        if not is_domain_relevant(
            clean_text,
            tokens,
            sentence_aspects,
            context_aspects,
            self.learned_relevance_terms,
        ):
            return self._output(clean_text, [])

        aspects = sentence_aspects
        if not aspects and context_aspects:
            aspects = [max(context_aspects, key=lambda item: item.priority)]
        if not aspects:
            aspects = self.aspect_detector.detect_implicit(clean_text, tokens)
        if not aspects:
            return self._output(clean_text, [])

        signals = self.sentiment_assigner.detect_signals(clean_text, tokens)
        clauses = segment_clauses(tokens)
        linked = self._link_aspects(aspects, signals, clauses, tokens, clean_text)

        quads: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in linked:
            strongest = self._strongest_evidence(item.evidence, item.aspect)
            opinion = strongest.text if strongest else ""
            sentiment = self.sentiment_assigner.polarity(
                item.score, self.sentiment_threshold
            )
            key = (
                item.aspect.normalized,
                item.category,
                opinion.casefold(),
                sentiment,
            )
            if key in seen:
                continue
            seen.add(key)
            quad: dict[str, Any] = {
                "aspect": item.aspect.text,
                "category": item.category,
                "opinion": opinion,
                "sentiment": sentiment,
            }
            if include_debug:
                quad["debug"] = {
                    "aspect_group": item.aspect.group,
                    "aspect_source": item.aspect.source,
                    "score": round(item.score, 4),
                    "confidence": round(item.confidence, 4),
                    "evidence": [
                        {
                            "text": signal.text,
                            "score": round(signal.score, 4),
                            "source": signal.source,
                            "negated": signal.negated,
                            "intensified": signal.intensified,
                        }
                        for signal in item.evidence
                    ],
                }
            quads.append(quad)
        return self._output(clean_text, quads)

    def predict_batch(
        self,
        records: Iterable[str | dict[str, str]],
        include_debug: bool = False,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for record in records:
            if isinstance(record, str):
                outputs.append(self.predict(record, include_debug=include_debug))
                continue
            outputs.append(
                self.predict(
                    record.get("text") or record.get("sentence") or "",
                    parent_context=record.get("parent_context", ""),
                    thread_title=record.get("thread_title", ""),
                    include_debug=include_debug,
                )
            )
        return outputs

    def __call__(
        self,
        text: str,
        parent_context: str = "",
        thread_title: str = "",
        include_debug: bool = False,
    ) -> dict[str, Any]:
        return self.predict(
            text,
            parent_context=parent_context,
            thread_title=thread_title,
            include_debug=include_debug,
        )

    @classmethod
    def from_artifact(cls, path: str | Path) -> "ABSASentimentModel":
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        thresholds = artifact.get("thresholds", {})
        return cls(
            max_link_distance=int(thresholds.get("max_link_distance", 12)),
            sentiment_threshold=float(thresholds.get("sentiment", 0.15)),
            learned_category_terms=artifact.get("learned_category_terms", {}),
            learned_aspect_terms=artifact.get("learned_aspect_terms", {}),
            learned_relevance_terms=artifact.get("learned_relevance_terms", []),
            promoted_sentiment_terms=artifact.get("promoted_sentiment_terms", {}),
            artifact_metadata=artifact.get("metadata", {}),
        )

    def detect_aspects(self, text: str) -> list[AspectCandidate]:
        clean_text, tokens = technical_tokenize(text)
        return self.aspect_detector.detect(clean_text, tokens)

    def detect_sentiment_signals(self, text: str) -> list[SentimentSignal]:
        clean_text, tokens = technical_tokenize(text)
        return self.sentiment_assigner.detect_signals(clean_text, tokens)

    def _context_aspects(
        self, parent_context: str, thread_title: str
    ) -> list[AspectCandidate]:
        candidates: list[AspectCandidate] = []
        for source, text in (
            ("parent_context", parent_context),
            ("thread_title", thread_title),
        ):
            clean_text, context_tokens = technical_tokenize(text)
            candidates.extend(
                self.aspect_detector.detect(clean_text, context_tokens, source=source)
            )
        deduped: dict[str, AspectCandidate] = {}
        for candidate in candidates:
            current = deduped.get(candidate.normalized)
            if current is None or candidate.priority > current.priority:
                deduped[candidate.normalized] = candidate
        return list(deduped.values())

    def _link_aspects(
        self,
        aspects: list[AspectCandidate],
        signals: list[SentimentSignal],
        clauses: list,
        tokens: list[TechnicalToken],
        text: str,
    ) -> list[LinkedAspect]:
        assigned: dict[int, list[SentimentSignal]] = defaultdict(list)
        sentence_aspect_indexes = [
            index
            for index, aspect in enumerate(aspects)
            if aspect.source == "sentence" and aspect.token_start >= 0
        ]

        for signal in signals:
            signal_clause = clause_index(signal.token_start, clauses)
            choices: list[tuple[int, int]] = []
            for index in sentence_aspect_indexes:
                aspect = aspects[index]
                aspect_clause = clause_index(aspect.token_start, clauses)
                if aspect_clause != signal_clause:
                    continue
                distance = self._token_distance(aspect, signal)
                if distance <= self.max_link_distance:
                    choices.append((distance, index))
            if choices:
                _, best_index = min(
                    choices,
                    key=lambda row: (row[0], -aspects[row[1]].priority),
                )
                assigned[best_index].append(signal)
            elif aspects and not sentence_aspect_indexes:
                assigned[0].append(signal)

        linked: list[LinkedAspect] = []
        for index, aspect in enumerate(aspects):
            evidence = tuple(assigned.get(index, []))
            score = self._aggregate(aspect, evidence)
            clause_text = self._clause_text(aspect, evidence, clauses, tokens, text)
            category = self.category_mapper.map(aspect, evidence, clause_text)
            evidence_strength = sum(abs(signal.score) for signal in evidence)
            source_discount = 0.65 if aspect.source != "sentence" else 1.0
            confidence = source_discount * (
                min(1.0, aspect.priority / 3.6) + min(1.0, evidence_strength / 2.0)
            ) / 2
            linked.append(
                LinkedAspect(
                    aspect=aspect,
                    evidence=evidence,
                    score=score,
                    confidence=confidence,
                    category=category,
                )
            )
        return linked

    @staticmethod
    def _token_distance(
        aspect: AspectCandidate, signal: SentimentSignal
    ) -> int:
        if aspect.token_end < signal.token_start:
            return signal.token_start - aspect.token_end
        if signal.token_end < aspect.token_start:
            return aspect.token_start - signal.token_end
        return 0

    def _aggregate(
        self, aspect: AspectCandidate, evidence: tuple[SentimentSignal, ...]
    ) -> float:
        if not evidence:
            return 0.0
        weighted_sum = 0.0
        total_weight = 0.0
        for signal in evidence:
            distance = (
                self._token_distance(aspect, signal)
                if aspect.token_start >= 0
                else 2
            )
            weight = 1.0 / (1.0 + 0.15 * distance)
            weighted_sum += signal.score * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight else 0.0

    def _strongest_evidence(
        self,
        evidence: tuple[SentimentSignal, ...],
        aspect: AspectCandidate,
    ) -> SentimentSignal | None:
        if not evidence:
            return None
        return max(
            evidence,
            key=lambda signal: (
                abs(signal.score),
                -self._token_distance(aspect, signal)
                if aspect.token_start >= 0
                else 0,
            ),
        )

    @staticmethod
    def _clause_text(
        aspect: AspectCandidate,
        evidence: tuple[SentimentSignal, ...],
        clauses: list,
        tokens: list[TechnicalToken],
        text: str,
    ) -> str:
        token_index = aspect.token_start
        if token_index < 0 and evidence:
            token_index = evidence[0].token_start
        target_clause = clause_index(token_index, clauses)
        if target_clause is None:
            return text
        clause = clauses[target_clause]
        if not tokens:
            return text
        start = tokens[clause.token_start].start
        end = tokens[clause.token_end].end
        return text[start:end]

    def _output(
        self, text: str, quads: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "text": text,
            "quads": quads,
            "model_name": self.MODEL_NAME,
        }


RuleBasedABSABaseline = ABSASentimentModel
