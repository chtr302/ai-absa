from dataclasses import dataclass


@dataclass(frozen=True)
class TechnicalToken:
    text: str
    normalized: str
    index: int
    start: int
    end: int


@dataclass(frozen=True)
class AspectCandidate:
    text: str
    normalized: str
    group: str
    start: int
    end: int
    priority: float
    token_start: int = -1
    token_end: int = -1
    source: str = "sentence"


@dataclass(frozen=True)
class SentimentSignal:
    text: str
    score: float
    start: int
    end: int
    token_start: int
    token_end: int
    source: str
    negated: bool = False
    intensified: bool = False

    @property
    def sentiment(self) -> str:
        if self.score > 0:
            return "Positive"
        if self.score < 0:
            return "Negative"
        return "Neutral"


@dataclass(frozen=True)
class Clause:
    index: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class LinkedAspect:
    aspect: AspectCandidate
    evidence: tuple[SentimentSignal, ...]
    score: float
    confidence: float
    category: str
