import json
import re
from pathlib import Path
from tqdm import tqdm
import random

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data/processed/filtering_data.jsonl"
OUTPUT_DIR = BASE_DIR / "data/processed/splits_v2"

TRAIN_SIZE = 20000
VAL_SIZE = 1000
TEST_SIZE = 1000

random.seed(42)

# =========================
# NORMALIZATION UTIL (FIXED)
# =========================
def normalize_text(t: str) -> str:
    t = t.lower()

    # remove special chars but keep spaces + numbers
    t = re.sub(r"[^a-z0-9\s]", " ", t)

    # collapse spaces
    t = re.sub(r"\s+", " ", t).strip()

    # gpt 4 -> gpt4 | grok 3 -> grok3 | qwen 3 -> qwen3
    t = re.sub(r"([a-z])\s+(\d+)", r"\1\2", t)

    return t


# =========================
# MODEL CANONICAL (EXPANDED FIXED)
# =========================
MODEL_CANONICAL = {
    "gpt": ["gpt", "chatgpt", "gpt3", "gpt4", "gpt4o", "gpt5"],
    "qwen": ["qwen", "qwen2", "qwen3", "qwen35", "qwq", "qwen coder"],
    "llama": ["llama", "llama2", "llama3", "llama31", "llama32", "llama cpp"],
    "mistral": ["mistral", "mixtral", "ministral"],
    "deepseek": ["deepseek", "deepseekr1", "deepseekv3", "deepseek coder"],
    "phi": ["phi", "phi3", "phi4"],
    "glm": ["glm", "glm4"],
    "gemini": ["gemini"],
    "claude": ["claude"],
    "grok": ["grok", "grok3", "grok4"],
    "codex": ["codex"],
    "openai": ["openai", "o1", "o3"]
}

# =========================
# HARDWARE CANONICAL (FIXED REAL WORLD)
# =========================
HARDWARE_CANONICAL = {
    "gpu": ["gpu", "rtx", "gtx", "3090", "4090", "5090", "a100", "h100", "titan", "tesla"],
    "cpu": ["cpu", "ryzen", "intel", "xeon", "epyc", "m2", "m3", "5800x", "i9"],
    "memory": ["ram", "vram", "ddr4", "ddr5", "hbm", "64gb", "128gb"],
    "accelerator": ["cuda", "rocm", "tpu", "vulkan", "onnx", "metal", "opencl"],
    "framework": ["vllm", "llama.cpp", "ollama", "kobold", "transformers"]
}

# =========================
# ENTITY EXTRACTOR (FIXED)
# =========================
def extract_entities(text: str):
    t = normalize_text(text)
    entities = []

    def match(variants, canon_type):
        for v in variants:
            if normalize_text(v) in t:
                entities.append({
                    "type": canon_type,
                    "value": v,          # giữ canonical raw type
                    "mention": v         # sẽ update sau nếu cần context
                })

    for canon, variants in MODEL_CANONICAL.items():
        match(variants, "model")

    for canon, variants in HARDWARE_CANONICAL.items():
        match(variants, "hardware")

    # dedup
    seen = set()
    out = []

    for e in entities:
        key = (e["type"], e["value"])
        if key not in seen:
            seen.add(key)
            out.append(e)

    return out


# =========================
# TECH SIGNALS
# =========================
TECH_KEYWORDS = [
    "llama", "qwen", "mistral", "gpt", "ollama",
    "vllm", "llama.cpp", "quant", "gguf",
    "gpu", "vram", "cuda", "inference", "training",
    "token", "benchmark", "latency", "throughput",
    "fp16", "fp8", "int4", "moe", "attention",
    "rocm", "huggingface", "transformer", "kv cache"
]

COMPARE_KEYWORDS = [
    "better", "faster", "vs", "compared", "than",
    "instead", "replace", "over", "underperform"
]

NOISE_PATTERNS = [
    "thank you", "thanks", "appreciate", "lol", "haha"
]


# =========================
# CLEAN
# =========================
def clean_text(text):
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?()\-/:]", " ", text)

    return text.strip()


def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]


def get_text_field(row):
    return row.get("body") or row.get("text") or row.get("content") or row.get("selftext") or ""


def compute_tech_score(text):
    t = text.lower()
    score = sum(1 for k in TECH_KEYWORDS if k in t)
    return min(score, 3)


def is_valid_sentence(text):
    t = text.lower()

    if any(p in t for p in NOISE_PATTERNS):
        return False

    has_tech = any(k in t for k in TECH_KEYWORDS)
    has_compare = any(k in t for k in COMPARE_KEYWORDS)

    return has_tech or has_compare


# =========================
# LOAD INDEX
# =========================
def load_index():
    body_map = {}
    title_map = {}

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except:
                continue

            rid = row.get("id")
            if not rid:
                continue

            body_map[rid] = get_text_field(row)
            title_map[rid] = row.get("title", "")

    return body_map, title_map


# =========================
# ANCHOR DETECT
# =========================
def detect_anchor(sentences):
    best_idx = 0
    best_score = -1

    for i, s in enumerate(sentences):
        t = s["text"].lower()
        score = compute_tech_score(t)

        if any(k in t for k in ["why", "problem", "issue", "fix", "error"]):
            score += 2

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


# =========================
# TRANSFORM (FIXED)
# =========================
def transform(row, body_map, title_map):
    raw_text = get_text_field(row)
    text = clean_text(raw_text)

    if len(text) < 20:
        return None

    sentences = split_sentences(text)
    sentences = [s for s in sentences if is_valid_sentence(s)]

    if not sentences:
        return None

    parent_id = row.get("parent_id", "")
    parent_context = clean_text(body_map.get(parent_id, ""))

    if not parent_context:
        parent_context = sentences[0][:300]

    thread_title = clean_text(title_map.get(parent_id, "")) or sentences[0][:120]

    sentence_blocks = []

    for s in sentences:
        sentence_blocks.append({
            "text": s,
            "is_anchor": False,
            "tech_score": float(compute_tech_score(s)),
            "entities": extract_entities(s),
            "quads": []
        })

    anchor_idx = detect_anchor(sentence_blocks)
    sentence_blocks[anchor_idx]["is_anchor"] = True

    return {
        "id": row.get("id"),
        "parent_id": parent_id,
        "parent_type": "submission" if parent_id.startswith("t3_") else "comment",
        "parent_context": parent_context,
        "thread_title": thread_title,
        "sentences": sentence_blocks
    }


# =========================
# PROCESS
# =========================
def process():
    print("Running data processing...")

    body_map, title_map = load_index()
    data = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            try:
                row = json.loads(line)
            except:
                continue

            item = transform(row, body_map, title_map)
            if item:
                data.append(item)

    return data


# =========================
# SPLIT
# =========================
def split_data(data):
    random.shuffle(data)

    test = data[:TEST_SIZE]
    val = data[TEST_SIZE:TEST_SIZE + VAL_SIZE]
    train = data[TEST_SIZE + VAL_SIZE:TEST_SIZE + VAL_SIZE + TRAIN_SIZE]
    pool = data[TEST_SIZE + VAL_SIZE + TRAIN_SIZE:]

    return train, val, test, pool


# =========================
# SAVE
# =========================
def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# =========================
# MAIN
# =========================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = process()

    print("Total samples:", len(data))

    train, val, test, pool = split_data(data)

    save_jsonl(OUTPUT_DIR / "train_v2.jsonl", train)
    save_jsonl(OUTPUT_DIR / "val_v2.jsonl", val)
    save_jsonl(OUTPUT_DIR / "test_v2.jsonl", test)
    save_jsonl(OUTPUT_DIR / "pool_v2.jsonl", pool)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)} | Pool: {len(pool)}")


if __name__ == "__main__":
    main()