from dataclasses import dataclass


@dataclass(frozen=True)
class AspectCandidate:
    text: str
    normalized: str
    group: str
    start: int
    end: int
    priority: float


@dataclass(frozen=True)
class SentimentSignal:
    text: str
    sentiment: str
    start: int
    end: int
    weight: float
    negated: bool = False
