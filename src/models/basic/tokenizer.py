import re

from .baseline_types import Clause, TechnicalToken
from .text_utils import normalize_text


_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[._/+:-][A-Za-z0-9]+)*(?:'[A-Za-z]+)?|[^\w\s]",
    re.UNICODE,
)
_CLAUSE_BREAK_WORDS = frozenset({"but", "however", "although", "while", "yet"})
_CLAUSE_BREAK_PUNCTUATION = frozenset({".", ";", "!", "?"})


def technical_tokenize(text: str) -> tuple[str, list[TechnicalToken]]:
    cleaned = normalize_text(text)
    tokens = [
        TechnicalToken(
            text=match.group(0),
            normalized=match.group(0).casefold(),
            index=index,
            start=match.start(),
            end=match.end(),
        )
        for index, match in enumerate(_TOKEN_RE.finditer(cleaned))
    ]
    return cleaned, tokens


def token_span_for_chars(
    tokens: list[TechnicalToken], start: int, end: int
) -> tuple[int, int]:
    covered = [token.index for token in tokens if token.start < end and token.end > start]
    if not covered:
        return -1, -1
    return covered[0], covered[-1]


def segment_clauses(tokens: list[TechnicalToken]) -> list[Clause]:
    if not tokens:
        return []

    clauses: list[Clause] = []
    start = 0
    for token in tokens:
        is_break = (
            token.text in _CLAUSE_BREAK_PUNCTUATION
            or token.normalized in _CLAUSE_BREAK_WORDS
        )
        if not is_break:
            continue
        end = max(start, token.index - 1)
        if end >= start:
            clauses.append(Clause(len(clauses), start, end))
        start = token.index + 1

    if start <= tokens[-1].index:
        clauses.append(Clause(len(clauses), start, tokens[-1].index))
    return clauses


def clause_index(token_index: int, clauses: list[Clause]) -> int | None:
    for clause in clauses:
        if clause.token_start <= token_index <= clause.token_end:
            return clause.index
    return None
