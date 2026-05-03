import json
import random
import re
from collections import defaultdict
from pathlib import Path

# keyword signals per sentiment class
_POS = re.compile(
    r'\b(good|better|best|great|amazing|awesome|excellent|perfect|love|fast|'
    r'recommend\w*|outperform\w*|superb|fantastic|impressive|solid|reliable)\b'
)
_NEG = re.compile(
    r'\b(bad|worse|worst|slow|issue|problem|bug|mistake|error|broken|fail\w*|'
    r'terrible|awful|garbage|hallucinat\w*|disappoint\w*|useless|annoying)\b'
)
_NEU = re.compile(
    r'\b(compare\w*|vs|difference|depends|both|try|gpu|vram|quant\w*|token|'
    r'context|ollama|endpoint|benchmark\w*|parameter\w*|inference|deploy\w*)\b'
)


def _assign_label(body: str) -> str | None:
    t = body.lower()
    scores = {
        "positive": len(_POS.findall(t)),
        "negative": len(_NEG.findall(t)),
        "neutral":  len(_NEU.findall(t)),
    }
    if not any(scores.values()):
        return None
    return max(scores, key=scores.get)


def load_labeled(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            body = (d.get("body") or "").strip()
            if not body or body in ("[removed]", "[deleted]"):
                continue
            if d.get("author") == "[deleted]":
                continue
            label = _assign_label(body)
            if label:
                records.append({"text": body, "label": label})
    return records


def balance(records: list[dict], n_per_class: int, seed: int = 42) -> list[dict]:
    """Undersample each class to n_per_class."""
    by_label: dict[str, list] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    rng = random.Random(seed)
    sampled: list[dict] = []
    for items in by_label.values():
        sampled.extend(rng.sample(items, min(n_per_class, len(items))))
    rng.shuffle(sampled)
    return sampled


def split(
    records: list[dict],
    dev_ratio: float = 0.70,
    train_ratio: float = 0.70,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Split records into train / val / test / leaderboard.
    dev_ratio   : fraction of total going to dev pool (70 %)
    train_ratio : fraction of dev pool going to train (70 % of 70 %)
    Returns (train, val, test, leaderboard)
    """
    by_label: dict[str, list] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    rng = random.Random(seed)
    dev_all: list[dict] = []
    leaderboard: list[dict] = []

    for items in by_label.values():
        rng.shuffle(items)
        cut = int(len(items) * dev_ratio)
        dev_all.extend(items[:cut])
        leaderboard.extend(items[cut:])

    # split dev into train / val / test  (70 / 15 / 15)
    by_label_dev: dict[str, list] = defaultdict(list)
    for r in dev_all:
        by_label_dev[r["label"]].append(r)

    trains, vals, tests = [], [], []
    val_ratio = (1 - train_ratio) / 2

    for items in by_label_dev.values():
        rng.shuffle(items)
        n = len(items)
        i1 = int(n * train_ratio)
        i2 = int(n * (train_ratio + val_ratio))
        trains.extend(items[:i1])
        vals.extend(items[i1:i2])
        tests.extend(items[i2:])

    rng.shuffle(trains)
    rng.shuffle(vals)
    rng.shuffle(tests)
    rng.shuffle(leaderboard)
    return trains, vals, tests, leaderboard
