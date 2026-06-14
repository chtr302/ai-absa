import re

from .baseline_types import AspectCandidate, TechnicalToken


_DOMAIN_RE = re.compile(
    r"\b(?:llm|ai|model|inference|quant(?:ization|ized)?|gguf|exl2|awq|vram|gpu|"
    r"cuda|rocm|ollama|vllm|llama\.cpp|context|tokens?|embedding|rag|lora|"
    r"prompt|benchmark|latency|throughput|fine[-\s]?tun(?:e|ing)|fp8|fp16|"
    r"bf16|kernel|marlin|ampere)\b",
    re.IGNORECASE,
)


def is_domain_relevant(
    text: str,
    tokens: list[TechnicalToken],
    sentence_aspects: list[AspectCandidate],
    context_aspects: list[AspectCandidate] | None = None,
    learned_terms: tuple[str, ...] = (),
) -> bool:
    del tokens  # Kept in the interface for later token-level relevance features.
    if sentence_aspects or context_aspects:
        return True
    normalized = text.casefold()
    if any(term in normalized for term in learned_terms):
        return True
    return bool(_DOMAIN_RE.search(text))
