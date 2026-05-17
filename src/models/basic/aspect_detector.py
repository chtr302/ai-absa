import re

from .baseline_lexicons import ASPECT_PATTERNS
from .baseline_types import AspectCandidate


class AspectDetector:
    def __init__(self, patterns: tuple[tuple[str, str, float], ...] = ASPECT_PATTERNS):
        self._aspect_res = tuple(
            (group, re.compile(self._bounded(pattern), re.IGNORECASE), priority)
            for group, pattern, priority in patterns
        )

    def detect(self, text: str) -> list[AspectCandidate]:
        candidates: list[AspectCandidate] = []
        for group, pattern, priority in self._aspect_res:
            for match in pattern.finditer(text):
                raw = match.group(0).strip()
                if not raw:
                    continue
                candidates.append(
                    AspectCandidate(
                        text=self._canonical_aspect(raw, group),
                        normalized=self._normalize_aspect(raw, group),
                        group=group,
                        start=match.start(),
                        end=match.end(),
                        priority=priority,
                    )
                )
        return self._dedupe_candidates(candidates)

    @staticmethod
    def _bounded(pattern: str) -> str:
        return rf"(?<![A-Za-z0-9_])(?:{pattern})(?![A-Za-z0-9_])"

    def _dedupe_candidates(self, candidates: list[AspectCandidate]) -> list[AspectCandidate]:
        candidates = sorted(candidates, key=lambda item: (item.start, -(item.end - item.start), -item.priority))
        kept: list[AspectCandidate] = []
        for candidate in candidates:
            if any(self._overlap(candidate, existing) for existing in kept):
                continue
            if any(candidate.normalized == existing.normalized for existing in kept):
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _overlap(left: AspectCandidate, right: AspectCandidate) -> bool:
        return max(left.start, right.start) < min(left.end, right.end)

    def _canonical_aspect(self, raw: str, group: str) -> str:
        raw = " ".join(raw.strip().split())
        lower = raw.lower()
        literal_map = {
            "vram": "VRAM",
            "gpu": "GPU",
            "ram": "RAM",
            "cuda": "CUDA",
            "rocm": "ROCm",
            "nvidia": "NVIDIA",
            "amd": "AMD",
            "gguf": "GGUF",
            "exl2": "EXL2",
            "awq": "AWQ",
            "fp16": "FP16",
            "bf16": "BF16",
            "rag": "RAG",
            "ollama": "Ollama",
            "llama.cpp": "llama.cpp",
            "vllm": "vLLM",
            "lm studio": "LM Studio",
            "koboldcpp": "KoboldCpp",
            "anythingllm": "AnythingLLM",
            "flash attention": "flash attention",
        }
        if lower in literal_map:
            return literal_map[lower]
        if lower.startswith("rtx"):
            return re.sub(r"\s+", " ", raw.upper())
        if group == "model":
            return self._canonical_model_name(raw)
        return raw

    @staticmethod
    def _canonical_model_name(raw: str) -> str:
        match = re.match(r"(?i)(gpt[-\s]?oss|chatgpt|deepseek|llama|qwen|gpt|claude|gemma|mistral|grok|phi|glm|yi)(.*)", raw)
        if not match:
            return raw
        prefix, suffix = match.groups()
        prefix_key = re.sub(r"[-\s]+", "", prefix.lower())
        prefix_map = {
            "gptoss": "GPT-OSS",
            "chatgpt": "ChatGPT",
            "deepseek": "DeepSeek",
            "llama": "Llama",
            "qwen": "Qwen",
            "gpt": "GPT",
            "claude": "Claude",
            "gemma": "Gemma",
            "mistral": "Mistral",
            "grok": "Grok",
            "phi": "Phi",
            "glm": "GLM",
            "yi": "Yi",
        }
        canonical_prefix = prefix_map.get(prefix_key, prefix)
        suffix = suffix.strip(" -_")
        if not suffix:
            return canonical_prefix
        separator = "-" if canonical_prefix in {"GPT", "GPT-OSS"} else " "
        return f"{canonical_prefix}{separator}{suffix.upper() if suffix.lower().endswith('b') else suffix}"

    def _normalize_aspect(self, raw: str, group: str) -> str:
        canonical = self._canonical_aspect(raw, group)
        return re.sub(r"\s+", "-", canonical.lower())
