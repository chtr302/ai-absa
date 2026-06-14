MODEL_NAME_PATTERNS: tuple[str, ...] = (
    r"gpt[-\s]?oss(?:[-\s]?\d+[bB]?)?",
    r"chatgpt",
    r"gpt(?:[-\s]?\d+(?:\.\d+)?[a-zA-Z]?)?",
    r"claude(?:[-\s]?\d+(?:\.\d+)?)?",
    r"deepseek(?:[-\s]?(?:r1|v\d+(?:\.\d+)?|coder|chat|distill))?",
    r"llama(?:[-\s]?\d+(?:\.\d+)?[a-zA-Z]?)?",
    r"qwen(?:[-\s]?\d+(?:\.\d+)?(?:[-\s]?(?:\d+[bB]|[aA]\d+[bB]|vl|coder|instruct|thinking))?)?",
    r"gemma(?:[-\s]?\d+(?:\.\d+)?[bB]?)?",
    r"mistral(?:[-\s]?(?:small|large|nemo|medium|codestral)(?:[-\s]?\d+(?:\.\d+)?)?)?",
    r"ministral(?:[-\s]?\d+(?:\.\d+)?[bB]?)?",
    r"devstral(?:[-\s]?(?:small|medium|\d+(?:\.\d+)?[bB]?))?",
    r"grok(?:[-\s]?\d+(?:\.\d+)?)?",
    r"phi(?:[-\s]?\d+(?:\.\d+)?)?",
    r"glm(?:[-\s]?\d+(?:\.\d+)?(?:[-\s]?(?:air|flash|plus))?)?",
    r"kimi(?:[-\s]?(?:k\d+(?:\.\d+)?|dev|vl))?",
    r"command[-\s]?[rR](?:[-\s]?plus)?",
    r"\byi\b",
)

FINAL_CATEGORIES: tuple[str, ...] = (
    "BEHAVIOR",
    "COMPARATIVE",
    "INTELLIGENCE",
    "PERFORMANCE",
    "RESOURCES",
    "SOFTWARE",
    "TECHNICAL",
)

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "COMPARATIVE": (
        "better", "best", "worse", "worst", "faster", "slower",
        "outperform", "superior", "inferior", "compared", "versus", " vs ",
        "than",
    ),
    "PERFORMANCE": (
        "fast", "slow", "latency", "throughput", "token/s", "tokens/s",
        "tokens per second", "t/s", "benchmark", "speed", "perplexity",
        "realtime", "real-time",
    ),
    "RESOURCES": (
        "vram", " ram", "memory", "gpu", "cpu", "cuda", "rocm", "power",
        "watt", "fit", "requires", "needs more", "uses more", "too much",
        "context window", "context length", "context limit",
    ),
    "SOFTWARE": (
        "ollama", "llama.cpp", "vllm", "lm studio", "koboldcpp",
        "anythingllm", "open webui", "framework", "api", "endpoint",
        "install", "support", "windows", "linux",
    ),
    "INTELLIGENCE": (
        "reason", "think", "smart", "coding", "code", "math", "knowledge",
        "accurate", "answer", "instruction", "tool calling", "function calling",
        "rag", "retrieval",
    ),
    "BEHAVIOR": (
        "hallucinat", "refus", "censor", "verbose", "repet", "lazy",
        "personality", "tone", "sycoph", "stable", "unstable", "crash",
        "forget", "memory of", "safe", "alignment",
    ),
    "TECHNICAL": (
        "quant", "gguf", "exl2", "awq", "fp8", "fp16", "bf16", "flash attention",
        "lora", "qlora", "fine-tun", "inference", "embedding", "prompt",
        "sampler", "temperature", "top_p", "top-k", "architecture", "weights",
        "kernel", "marlin", "ampere",
    ),
}

NEGATIONS = frozenset(
    {
        "not", "never", "no", "cannot", "can't", "cant", "don't", "dont",
        "doesn't", "doesnt", "isn't", "isnt", "wasn't", "wasnt", "hardly",
        "barely",
    }
)

INTENSIFIERS: dict[str, float] = {
    "very": 1.25,
    "really": 1.25,
    "extremely": 1.5,
    "incredibly": 1.4,
    "pretty": 1.15,
    "quite": 1.15,
    "massively": 1.4,
    "super": 1.3,
}

STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "for", "from", "had", "has", "have", "he", "her", "his", "i", "if",
        "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "she",
        "so", "that", "the", "their", "them", "they", "this", "to", "was", "we",
        "were", "will", "with", "you", "your",
    }
)

IMPLICIT_MARKERS = ("this model", "the model", "this", "it", "they")
