import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from src.data.dataset import BioSequenceSample, bio_tags_to_token_spans


DEFAULT_TAGS = ("B-ASP", "I-ASP", "O")
UNKNOWN_TOKEN = "<UNK>"
NEG_INF = float("-inf")


def _is_legal_start(tag: str) -> bool:
    return tag != "I-ASP"


def _is_legal_transition(previous_tag: str, current_tag: str) -> bool:
    return not (previous_tag == "O" and current_tag == "I-ASP")


class HiddenMarkovAspectExtractor:
    def __init__(
        self,
        tags: Sequence[str] = DEFAULT_TAGS,
        smoothing: float = 1.0,
        min_token_freq: int = 1,
    ) -> None:
        self.tags = list(tags)
        self.smoothing = smoothing
        self.min_token_freq = min_token_freq
        self.vocabulary: set[str] = set()
        self.start_log_probs: Dict[str, float] = {}
        self.transition_log_probs: Dict[str, Dict[str, float]] = {}
        self.emission_log_probs: Dict[str, Dict[str, float]] = {}
        self.fitted = False

    def fit(self, samples: Sequence[BioSequenceSample]) -> None:
        if not samples:
            raise ValueError("Cannot train HMM with an empty sample list.")

        token_frequencies = Counter(
            self._normalize_token(token)
            for sample in samples
            for token in sample.tokens
        )
        self.vocabulary = {
            token for token, count in token_frequencies.items() if count >= self.min_token_freq
        }

        start_counts = Counter({tag: 0 for tag in self.tags})
        transition_counts = {
            tag: Counter({next_tag: 0 for next_tag in self.tags}) for tag in self.tags
        }
        emission_counts = {
            tag: Counter({UNKNOWN_TOKEN: 0}) for tag in self.tags
        }
        state_counts = Counter({tag: 0 for tag in self.tags})

        for sample in samples:
            if not sample.tokens:
                continue
            tags = self._normalize_bio_tags(sample.bio_tags)
            start_counts[tags[0]] += 1

            for index, (token, tag) in enumerate(zip(sample.tokens, tags)):
                observation = self._encode_observation(token)
                emission_counts[tag][observation] += 1
                state_counts[tag] += 1
                if index > 0:
                    transition_counts[tags[index - 1]][tag] += 1

        num_sequences = sum(1 for sample in samples if sample.tokens)
        self.start_log_probs = self._build_log_distribution(
            counts=start_counts,
            categories=self.tags,
            total_override=num_sequences,
        )
        self.transition_log_probs = {
            tag: self._build_log_distribution(transition_counts[tag], self.tags)
            for tag in self.tags
        }

        observation_space = sorted(self.vocabulary | {UNKNOWN_TOKEN})
        self.emission_log_probs = {
            tag: self._build_log_distribution(
                counts=emission_counts[tag],
                categories=observation_space,
                total_override=state_counts[tag],
            )
            for tag in self.tags
        }
        self.fitted = True

    def predict(self, tokens: Sequence[str]) -> List[str]:
        if not self.fitted:
            raise RuntimeError("Model must be trained or loaded before prediction.")
        if not tokens:
            return []

        observations = [self._encode_observation(token) for token in tokens]
        scores: List[Dict[str, float]] = []
        backpointers: List[Dict[str, str | None]] = []

        first_scores: Dict[str, float] = {}
        first_backpointers: Dict[str, str | None] = {}
        for tag in self.tags:
            if not _is_legal_start(tag):
                first_scores[tag] = NEG_INF
                first_backpointers[tag] = None
                continue
            first_scores[tag] = self.start_log_probs[tag] + self._emission_log_prob(tag, observations[0])
            first_backpointers[tag] = None
        scores.append(first_scores)
        backpointers.append(first_backpointers)

        for index in range(1, len(observations)):
            current_scores: Dict[str, float] = {}
            current_backpointers: Dict[str, str | None] = {}
            for current_tag in self.tags:
                best_previous_tag = None
                best_score = NEG_INF
                emission_score = self._emission_log_prob(current_tag, observations[index])

                for previous_tag in self.tags:
                    if not _is_legal_transition(previous_tag, current_tag):
                        continue
                    previous_score = scores[index - 1][previous_tag]
                    if previous_score == NEG_INF:
                        continue
                    candidate_score = (
                        previous_score
                        + self.transition_log_probs[previous_tag][current_tag]
                        + emission_score
                    )
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_previous_tag = previous_tag

                current_scores[current_tag] = best_score
                current_backpointers[current_tag] = best_previous_tag

            scores.append(current_scores)
            backpointers.append(current_backpointers)

        last_tag = max(self.tags, key=lambda tag: scores[-1][tag])
        predicted_tags = [last_tag]
        for index in range(len(tokens) - 1, 0, -1):
            previous_tag = backpointers[index][predicted_tags[-1]]
            if previous_tag is None:
                previous_tag = "O"
            predicted_tags.append(previous_tag)

        predicted_tags.reverse()
        return self._normalize_bio_tags(predicted_tags)

    def predict_batch(self, samples: Sequence[BioSequenceSample]) -> List[List[str]]:
        return [self.predict(sample.tokens) for sample in samples]

    def extract_aspects(self, tokens: Sequence[str]) -> List[Dict[str, object]]:
        predicted_tags = self.predict(tokens)
        spans = bio_tags_to_token_spans(predicted_tags)
        return [
            {
                "start": start,
                "end": end,
                "text": " ".join(tokens[start:end]),
            }
            for start, end in spans
        ]

    def save(self, output_path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted HMM model.")

        payload = {
            "tags": self.tags,
            "smoothing": self.smoothing,
            "min_token_freq": self.min_token_freq,
            "vocabulary": sorted(self.vocabulary),
            "start_log_probs": self.start_log_probs,
            "transition_log_probs": self.transition_log_probs,
            "emission_log_probs": self.emission_log_probs,
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_path: str | Path) -> "HiddenMarkovAspectExtractor":
        with Path(model_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        model = cls(
            tags=payload["tags"],
            smoothing=payload["smoothing"],
            min_token_freq=payload["min_token_freq"],
        )
        model.vocabulary = set(payload["vocabulary"])
        model.start_log_probs = {
            str(tag): float(value) for tag, value in payload["start_log_probs"].items()
        }
        model.transition_log_probs = {
            str(tag): {str(next_tag): float(value) for next_tag, value in values.items()}
            for tag, values in payload["transition_log_probs"].items()
        }
        model.emission_log_probs = {
            str(tag): {str(token): float(value) for token, value in values.items()}
            for tag, values in payload["emission_log_probs"].items()
        }
        model.fitted = True
        return model

    def _normalize_token(self, token: str) -> str:
        return token.lower()

    def _encode_observation(self, token: str) -> str:
        normalized = self._normalize_token(token)
        if normalized in self.vocabulary:
            return normalized
        return UNKNOWN_TOKEN

    def _emission_log_prob(self, tag: str, observation: str) -> float:
        tag_distribution = self.emission_log_probs[tag]
        return tag_distribution.get(observation, tag_distribution[UNKNOWN_TOKEN])

    def _build_log_distribution(
        self,
        counts: Counter[str],
        categories: Iterable[str],
        total_override: int | None = None,
    ) -> Dict[str, float]:
        categories = list(categories)
        denominator = (
            total_override if total_override is not None else sum(counts.values())
        ) + (self.smoothing * len(categories))
        return {
            category: math.log((counts[category] + self.smoothing) / denominator)
            for category in categories
        }

    def _normalize_bio_tags(self, tags: Sequence[str]) -> List[str]:
        normalized: List[str] = []
        previous = "O"
        for tag in tags:
            current = tag if tag in self.tags else "O"
            if current == "I-ASP" and previous == "O":
                current = "B-ASP"
            normalized.append(current)
            previous = current
        return normalized
