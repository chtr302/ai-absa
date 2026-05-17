import re

from .baseline_lexicons import NEGATIVE_PATTERNS, POSITIVE_PATTERNS
from .baseline_types import AspectCandidate, SentimentSignal


class SentimentAssigner:
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"

    _NEGATION_RE = re.compile(
        r"\b(?:not|never|no|cannot|can't|dont|don't|doesnt|doesn't|isnt|isn't|wasnt|wasn't)\b|n't",
        re.IGNORECASE,
    )

    def __init__(
        self,
        positive_patterns: tuple[tuple[str, float], ...] = POSITIVE_PATTERNS,
        negative_patterns: tuple[tuple[str, float], ...] = NEGATIVE_PATTERNS,
    ):
        self._positive_res = tuple(
            (re.compile(pattern, re.IGNORECASE), weight)
            for pattern, weight in positive_patterns
        )
        self._negative_res = tuple(
            (re.compile(pattern, re.IGNORECASE), weight)
            for pattern, weight in negative_patterns
        )

    def detect_signals(self, text: str) -> list[SentimentSignal]:
        signals: list[SentimentSignal] = []
        for pattern, weight in self._positive_res:
            for match in pattern.finditer(text):
                sentiment = self.POSITIVE
                negated = self._is_negated(text, match.start())
                if negated:
                    sentiment = self.NEGATIVE
                signals.append(
                    SentimentSignal(match.group(0), sentiment, match.start(), match.end(), weight, negated)
                )

        for pattern, weight in self._negative_res:
            for match in pattern.finditer(text):
                sentiment = self.NEGATIVE
                negated = self._is_negated(text, match.start())
                if negated:
                    sentiment = self.POSITIVE
                signals.append(
                    SentimentSignal(match.group(0), sentiment, match.start(), match.end(), weight, negated)
                )

        return self._dedupe_signals(signals)

    def select_primary_aspect(
        self,
        candidates: list[AspectCandidate],
        signals: list[SentimentSignal],
        text: str,
    ) -> tuple[AspectCandidate | None, float, SentimentSignal | None]:
        if not candidates:
            return None, 0.0, None

        if not signals:
            selected = max(candidates, key=lambda item: (item.priority, -item.start))
            return selected, selected.priority, None

        best_candidate: AspectCandidate | None = None
        best_signal: SentimentSignal | None = None
        best_score = -1.0
        for candidate in candidates:
            for signal in signals:
                score = self.candidate_signal_score(candidate, signal, text)
                if score > best_score:
                    best_candidate = candidate
                    best_signal = signal
                    best_score = score
        return best_candidate, best_score, best_signal

    def assign_sentiment(
        self,
        candidate: AspectCandidate,
        signals: list[SentimentSignal],
        text: str,
    ) -> str:
        if not signals:
            return self.NEUTRAL

        best_signal = max(signals, key=lambda signal: self.candidate_signal_score(candidate, signal, text))
        return best_signal.sentiment

    def candidate_signal_score(self, candidate: AspectCandidate, signal: SentimentSignal, text: str) -> float:
        distance = self._distance(candidate, signal)
        proximity = max(0.0, 1.0 - (distance / 120.0))
        score = candidate.priority + signal.weight + (1.8 * proximity)

        if self._same_simple_clause(candidate, signal, text):
            score += 0.5
        else:
            score -= 0.4

        if self._is_comparative_signal(signal.text):
            if candidate.end <= signal.start:
                score += 0.9
            elif candidate.start >= signal.end:
                score -= 0.8

        if candidate.group == "hardware" and self._is_resource_usage(signal.text):
            score += 0.8

        return score

    @staticmethod
    def _distance(candidate: AspectCandidate, signal: SentimentSignal) -> int:
        if candidate.end <= signal.start:
            return signal.start - candidate.end
        if signal.end <= candidate.start:
            return candidate.start - signal.end
        return 0

    @staticmethod
    def _same_simple_clause(candidate: AspectCandidate, signal: SentimentSignal, text: str) -> bool:
        left = min(candidate.end, signal.end)
        right = max(candidate.start, signal.start)
        between = text[left:right].lower()
        return not re.search(r"[.;!?]|\bbut\b|\bhowever\b|\balthough\b|\bwhile\b", between)

    @staticmethod
    def _is_comparative_signal(signal_text: str) -> bool:
        return bool(re.search(r"\b(?:better|faster|worse|slower|superior|outperform)\b", signal_text, re.IGNORECASE))

    @staticmethod
    def _is_resource_usage(signal_text: str) -> bool:
        return bool(re.search(r"\b(?:too\s+much|uses?\s+more|needs?\s+more|memory\s+hog)\b", signal_text, re.IGNORECASE))

    def _is_negated(self, text: str, start: int) -> bool:
        prefix = text[max(0, start - 35):start]
        return bool(self._NEGATION_RE.search(prefix))

    def _dedupe_signals(self, signals: list[SentimentSignal]) -> list[SentimentSignal]:
        signals = sorted(signals, key=lambda item: (item.start, -(item.end - item.start), -item.weight))
        kept: list[SentimentSignal] = []
        for signal in signals:
            if any(self._signal_overlap(signal, existing) for existing in kept):
                continue
            kept.append(signal)
        return kept

    @staticmethod
    def _signal_overlap(left: SentimentSignal, right: SentimentSignal) -> bool:
        return max(left.start, right.start) < min(left.end, right.end)
