"""Small model catalog with aliases retained from the original CLI."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    runtime: str = "transformers"
    supports_chat: bool = True
    params_billion: float | None = None


class ModelRegistry:
    _MODEL_ID_PATTERN = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
    )
    ALIASES = {
        "gemma_small": ModelSpec(
            "google/gemma-3-1b-it", "Gemma 3 1B", params_billion=1.0
        ),
        "gemma": ModelSpec(
            "google/gemma-3-4b-it", "Gemma 3 4B", params_billion=4.3
        ),
        "gemma_4": ModelSpec(
            "google/gemma-4-26B-A4B-it", "Gemma 4 26B A4B", params_billion=26.0
        ),
        "code_gemma": ModelSpec(
            "google/codegemma-1.1-2b",
            "CodeGemma 2B",
            supports_chat=False,
            params_billion=2.5,
        ),
        "code_gemma_it": ModelSpec(
            "google/codegemma-1.1-7b-it",
            "CodeGemma 7B Instruct",
            params_billion=7.0,
        ),
        "llama": ModelSpec(
            "meta-llama/Llama-3.2-3B-Instruct",
            "Llama 3.2 3B Instruct",
            params_billion=3.2,
        ),
        "llama_small": ModelSpec(
            "meta-llama/Llama-3.2-1B-Instruct",
            "Llama 3.2 1B Instruct",
            params_billion=1.24,
        ),
        "qwen3-8b": ModelSpec(
            "Qwen/Qwen3-8B", "Qwen3 8B", params_billion=8.2
        ),
    }

    def resolve(self, model_name: str) -> ModelSpec:
        key = model_name.strip().casefold()
        if key in self.ALIASES:
            return self.ALIASES[key]
        model_id = model_name.strip()
        for spec in self.ALIASES.values():
            if spec.model_id == model_id:
                return spec
        if model_id.count("/") == 1 and self._MODEL_ID_PATTERN.fullmatch(model_id):
            return ModelSpec(model_id, model_id)
        supported = ", ".join(sorted(self.ALIASES))
        raise ValueError(
            f"Unknown model alias {model_name!r}. Use a Hugging Face model ID "
            f"or one of: {supported}"
        )

    def list_models(self) -> list[ModelSpec]:
        return list(self.ALIASES.values())
