from typing import Any, Iterable

from .aspect_detector import AspectDetector
from .baseline_types import AspectCandidate, SentimentSignal
from .sentiment_rules import SentimentAssigner
from .text_utils import normalize_text


class ABSASentimentModel:
    """
    Simple rule-based ABSA baseline.

    The model intentionally predicts at most one aspect-level sentiment for one
    raw sentence. This makes it conservative enough for noisy Reddit comments
    where aspect relations are often implicit or irregular.
    """

    POSITIVE = SentimentAssigner.POSITIVE
    NEGATIVE = SentimentAssigner.NEGATIVE
    NEUTRAL = SentimentAssigner.NEUTRAL

    def __init__(self, min_prediction_confidence: float = 1.7):
        self.min_prediction_confidence = min_prediction_confidence
        self.aspect_detector = AspectDetector()
        self.sentiment_assigner = SentimentAssigner()

    def predict(self, text: str, include_debug: bool = False) -> dict[str, Any]:
        """
        Predict one primary aspect sentiment from one raw sentence.

        Default output intentionally contains only the baseline core fields:
        aspect and sentiment. Debug fields are opt-in.
        """
        clean_text = self.normalize_text(text)
        candidates = self.detect_aspects(clean_text)
        if not clean_text or not candidates:
            return {"text": clean_text, "predictions": []}

        signals = self.detect_sentiment_signals(clean_text)
        selected, confidence, signal = self.select_primary_aspect(candidates, signals, clean_text)
        if selected is None or confidence < self.min_prediction_confidence:
            return {"text": clean_text, "predictions": []}

        sentiment = self.assign_sentiment(selected, signals, clean_text)
        prediction: dict[str, Any] = {
            "aspect": selected.text,
            "sentiment": sentiment,
        }
        if include_debug:
            prediction.update(
                {
                    "normalized_aspect": selected.normalized,
                    "aspect_group": selected.group,
                    "confidence": round(confidence, 3),
                }
            )
            if signal is not None:
                prediction.update(
                    {
                        "evidence": signal.text,
                        "evidence_sentiment": signal.sentiment,
                        "negated": signal.negated,
                    }
                )

        return {"text": clean_text, "predictions": [prediction]}

    def __call__(self, text: str, include_debug: bool = False) -> dict[str, Any]:
        return self.predict(text, include_debug=include_debug)

    def predict_batch(self, texts: Iterable[str], include_debug: bool = False) -> list[dict[str, Any]]:
        return [self.predict(text, include_debug=include_debug) for text in texts]

    @staticmethod
    def normalize_text(text: str) -> str:
        return normalize_text(text)

    def detect_aspects(self, text: str) -> list[AspectCandidate]:
        return self.aspect_detector.detect(text)

    def detect_sentiment_signals(self, text: str) -> list[SentimentSignal]:
        return self.sentiment_assigner.detect_signals(text)

    def select_primary_aspect(
        self,
        candidates: list[AspectCandidate],
        signals: list[SentimentSignal],
        text: str,
    ) -> tuple[AspectCandidate | None, float, SentimentSignal | None]:
        return self.sentiment_assigner.select_primary_aspect(candidates, signals, text)

    def assign_sentiment(
        self,
        candidate: AspectCandidate,
        signals: list[SentimentSignal],
        text: str,
    ) -> str:
        return self.sentiment_assigner.assign_sentiment(candidate, signals, text)


RuleBasedABSABaseline = ABSASentimentModel
