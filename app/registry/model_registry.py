"""Small model catalog with aliases retained from the original CLI."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    runtime: str = "transformers"


class ModelRegistry:
    _MODEL_ID_PATTERN = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
    )
    ALIASES = {
        "gemma_small": ModelSpec("google/gemma-3-1b-it", "Gemma 3 1B"),
        "gemma": ModelSpec("google/gemma-3-4b-it", "Gemma 3 4B"),
        "gemma_4": ModelSpec("google/gemma-4-26B-A4B-it", "Gemma 4 26B A4B"),
        "code_gemma": ModelSpec("google/codegemma-1.1-2b", "CodeGemma 2B"),
        "llama": ModelSpec(
            "meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B Instruct"
        ),
        "llama_small": ModelSpec(
            "meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B Instruct"
        ),
        "qwen3-8b": ModelSpec("Qwen/Qwen3-8B", "Qwen3 8B"),
    }

    def resolve(self, model_name: str) -> ModelSpec:
        key = model_name.strip().casefold()
        if key in self.ALIASES:
            return self.ALIASES[key]
        model_id = model_name.strip()
        if model_id.count("/") == 1 and self._MODEL_ID_PATTERN.fullmatch(model_id):
            return ModelSpec(model_id, model_id)
        supported = ", ".join(sorted(self.ALIASES))
        raise ValueError(
            f"Unknown model alias {model_name!r}. Use a Hugging Face model ID "
            f"or one of: {supported}"
        )

    def list_models(self) -> list[ModelSpec]:
        return list(self.ALIASES.values())
