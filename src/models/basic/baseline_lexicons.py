from .constants import MODEL_NAME_PATTERNS


ASPECT_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("software", r"llama\.cpp", 3.6),
    ("software", r"lm\s+studio", 3.4),
    ("software", r"text-generation-webui", 3.3),
    ("software", r"open\s+webui", 3.3),
    ("software", r"koboldcpp", 3.2),
    ("software", r"anythingllm", 3.2),
    ("software", r"ollama", 3.2),
    ("software", r"vllm", 3.2),
    *(("model", pattern, 3.4) for pattern in MODEL_NAME_PATTERNS),
    ("resources", r"rtx\s*\d{4}(?:\s*ti)?", 3.0),
    ("resources", r"\b(?:3090|4090|5070|5090)\s*(?:ti)?\b", 2.9),
    ("resources", r"\b(?:gpu|vram|ram|cpu|cuda|rocm|nvidia|amd)\b", 2.8),
    ("resources", r"\bmemory\s+(?:use|usage|requirement|requirements|footprint)\b", 3.1),
    ("resources", r"\bcontext\s+(?:length|window|limit)\b", 2.9),
    ("performance", r"\b(?:latency|throughput|perplexity|speed|benchmark)\b", 2.9),
    ("performance", r"tokens?\s*(?:/|per)\s*sec(?:ond)?", 3.0),
    ("performance", r"\bt/s\b", 3.0),
    ("behavior", r"\b(?:hallucination|hallucinations|refusal|refusals|censorship|verbosity)\b", 2.8),
    ("intelligence", r"\b(?:reasoning|coding|knowledge|math|accuracy|tool\s+calling)\b", 2.7),
    ("technical", r"\bq\d(?:_[a-z0-9]+)+\b", 2.8),
    ("technical", r"\b(?:gguf|exl2|awq|fp8|fp16|bf16|kv\s+cache)\b", 2.8),
    ("technical", r"\b(?:kernel|marlin|ampere)\b", 2.5),
    ("technical", r"\b(?:quantization|quantized|quantize|inference|embedding|prompt|sampler)\b", 2.6),
    ("technical", r"flash\s+attention", 2.8),
    ("technical", r"\b(?:lora|qlora|fine[-\s]?tune|fine[-\s]?tuning)\b", 2.6),
)


# Phrase patterns are intentionally listed before word patterns. The detector
# sorts matches by token length again before resolving overlap.
POSITIVE_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"\bless\s+hallucinat(?:e|es|ion|ions|ing)?\b", 1.7),
    (r"\bworks?\s+(?:really\s+|very\s+)?well\b", 1.5),
    (r"\bsweet\s+spot\b", 1.4),
    (r"\b(?:fit|fits)\s+(?:on|in)\b", 1.2),
    (r"\boutperform(?:s|ed|ing)?\b", 1.4),
    (r"\b(?:good|great|excellent|amazing|impressive|useful|solid|decent)\b", 1.0),
    (r"\b(?:fast|faster|best|better|accurate|smart|smooth|cheap|efficient|stable)\b", 1.0),
    (r"\b(?:recommend|love|superior|reliable)\b", 1.0),
)

NEGATIVE_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"\bnot\s+(?:good|great|accurate|stable|reliable|useful)\b", -1.5),
    (r"\bincorrect\s+(?:answer|answers|output)\b", -1.5),
    (r"\bout\s+of\s+memory\b", -1.8),
    (r"\btoo\s+much\b", -1.5),
    (r"\bmemory\s+hog\b", -1.6),
    (r"\buses?\s+more\b", -1.2),
    (r"\bneeds?\s+more\b", -1.2),
    (r"\b(?:bad|terrible|poor|slow|slower|worse|expensive|buggy|unstable|broken|trash)\b", -1.0),
    (r"\b(?:fail|fails|failed|failing|hallucinate|hallucinates|hallucinated|dislike)\b", -1.0),
    (r"\b(?:crash|crashes|crashed|censored|refuses?|useless|annoying)\b", -1.0),
)
