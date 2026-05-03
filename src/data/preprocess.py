import re

_NEGATIONS = frozenset({
    "not", "no", "never", "neither", "nor",
    "cannot", "cant", "cant", "wont", "dont", "doesnt",
    "isnt", "wasnt", "arent", "werent", "hardly", "barely",
})

# markdown code blocks, URLs, non-alpha chars
_RE_CODE  = re.compile(r'```[\s\S]*?```')
_RE_URL   = re.compile(r'http\S+|www\S+')
_RE_ALPHA = re.compile(r'[^a-z\s]')
_RE_WS    = re.compile(r'\s+')


def clean(text: str) -> str:
    text = text.lower()
    text = _RE_CODE.sub(' ', text)
    text = _RE_URL.sub(' ', text)
    text = _RE_ALPHA.sub(' ', text)
    return _RE_WS.sub(' ', text).strip()


def _negate(tokens: list[str], window: int = 3) -> list[str]:
    out: list[str] = []
    neg_count = 0
    in_neg = False
    for tok in tokens:
        if tok in _NEGATIONS:
            in_neg, neg_count = True, 0
            out.append(tok)
        elif in_neg:
            out.append(f"NOT_{tok}")
            neg_count += 1
            if neg_count >= window:
                in_neg = False
        else:
            out.append(tok)
    return out


def preprocess(text: str) -> str:
    tokens = clean(text).split()
    return " ".join(_negate(tokens))
