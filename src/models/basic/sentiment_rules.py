import re

from .baseline_lexicons import NEGATIVE_PATTERNS, POSITIVE_PATTERNS
from .baseline_types import SentimentSignal, TechnicalToken
from .constants import INTENSIFIERS, NEGATIONS
from .tokenizer import token_span_for_chars


class SentimentAssigner:
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"

    def __init__(
        self,
        positive_patterns: tuple[tuple[str, float], ...] = POSITIVE_PATTERNS,
        negative_patterns: tuple[tuple[str, float], ...] = NEGATIVE_PATTERNS,
        promoted_patterns: tuple[tuple[str, float], ...] = (),
    ):
        patterns = positive_patterns + negative_patterns + promoted_patterns
        self._signal_res = tuple(
            (re.compile(pattern, re.IGNORECASE), score)
            for pattern, score in patterns
        )

    def detect_signals(
        self, text: str, tokens: list[TechnicalToken]
    ) -> list[SentimentSignal]:
        raw_matches: list[tuple[int, int, re.Match[str], float]] = []
        for pattern, score in self._signal_res:
            for match in pattern.finditer(text):
                token_start, token_end = token_span_for_chars(
                    tokens, match.start(), match.end()
                )
                length = token_end - token_start + 1 if token_start >= 0 else 0
                raw_matches.append((length, len(match.group(0)), match, score))

        raw_matches.sort(key=lambda row: (-row[0], -row[1], row[2].start()))
        occupied: set[int] = set()
        signals: list[SentimentSignal] = []
        sarcasm = bool(re.search(r"(?:^|\s)/s(?:\s|$)", text, re.IGNORECASE))

        for _, _, match, base_score in raw_matches:
            token_start, token_end = token_span_for_chars(
                tokens, match.start(), match.end()
            )
            if token_start < 0:
                continue
            span_tokens = set(range(token_start, token_end + 1))
            if occupied & span_tokens:
                continue

            score, negated, intensified = self._apply_modifiers(
                base_score, token_start, tokens
            )
            if sarcasm and score > 0:
                score = -score * 0.7
            signals.append(
                SentimentSignal(
                    text=match.group(0),
                    score=score,
                    start=match.start(),
                    end=match.end(),
                    token_start=token_start,
                    token_end=token_end,
                    source="phrase" if token_end > token_start else "word",
                    negated=negated,
                    intensified=intensified,
                )
            )
            occupied.update(span_tokens)

        return sorted(signals, key=lambda signal: signal.start)

    @staticmethod
    def polarity(score: float, threshold: float = 0.15) -> str:
        if score > threshold:
            return SentimentAssigner.POSITIVE
        if score < -threshold:
            return SentimentAssigner.NEGATIVE
        return SentimentAssigner.NEUTRAL

    @staticmethod
    def _apply_modifiers(
        score: float, token_start: int, tokens: list[TechnicalToken]
    ) -> tuple[float, bool, bool]:
        prefix = tokens[max(0, token_start - 3):token_start]
        negated = any(token.normalized in NEGATIONS or token.normalized.endswith("n't") for token in prefix)
        multiplier = 1.0
        intensified = False
        for token in prefix:
            if token.normalized in INTENSIFIERS:
                multiplier *= INTENSIFIERS[token.normalized]
                intensified = True
        score *= multiplier
        if negated:
            score *= -0.85
        return score, negated, intensified
