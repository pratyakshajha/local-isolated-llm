"""Popular Hugging Face text-generation models with a local fallback."""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from time import monotonic
from urllib.parse import quote, urlencode
from urllib.request import urlopen

from app.registry.model_registry import ModelRegistry

_PUBLISHERS = ("google", "meta-llama", "Qwen", "deepseek-ai", "mistralai")
_HUB_API = "https://huggingface.co/api/models"
_CACHE_SECONDS = 3600
_REQUEST_TIMEOUT = 8
_cached_at = 0.0
_cached_catalog: tuple[list[dict], str] | None = None


def _get_json(url: str):
    with urlopen(url, timeout=_REQUEST_TIMEOUT) as response:
        return json.load(response)


def _parameters_billion(model: dict) -> float | None:
    safetensors = model.get("safetensors")
    if not isinstance(safetensors, dict):
        return None
    total = safetensors.get("total")
    if total is None and isinstance(safetensors.get("parameters"), dict):
        total = sum(safetensors["parameters"].values())
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return None
    if not math.isfinite(total) or total <= 0:
        return None
    return round(total / 1_000_000_000, 2)


def _download_gb(model_id: str) -> float | None:
    details = _get_json(f"{_HUB_API}/{quote(model_id, safe='/')}?blobs=true")
    siblings = details.get("siblings") if isinstance(details, dict) else None
    if not isinstance(siblings, list):
        return None
    sizes = []
    for file in siblings:
        filename = file.get("rfilename") if isinstance(file, dict) else None
        if not isinstance(filename, str) or not filename.endswith(".safetensors"):
            continue
        size = file.get("size")
        if size is None and isinstance(file.get("lfs"), dict):
            size = file["lfs"].get("size")
        if isinstance(size, bool) or not isinstance(size, (int, float)) or size < 0:
            return None
        sizes.append(size)
    return None if not sizes else sum(sizes) / 1_000_000_000


def _hub_catalog() -> list[dict]:
    registry = ModelRegistry()
    candidates: dict[str, dict] = {}
    for publisher in _PUBLISHERS:
        query = urlencode(
            [
                ("author", publisher),
                ("pipeline_tag", "text-generation"),
                ("sort", "downloads"),
                ("direction", "-1"),
                ("limit", "10"),
                ("expand", "safetensors"),
            ]
        )
        response = _get_json(f"{_HUB_API}?{query}")
        if not isinstance(response, list):
            raise ValueError("Unexpected Hugging Face model list response")
        for model in response:
            if not isinstance(model, dict):
                continue
            model_id = model.get("id")
            if (
                not isinstance(model_id, str)
                or model_id.split("/", 1)[0] != publisher
                or model_id in candidates
            ):
                continue
            try:
                registry.resolve(model_id)
            except ValueError:
                continue
            params_billion = _parameters_billion(model)
            if params_billion is None:
                continue
            candidates[model_id] = {
                "id": model_id,
                "name": model_id,
                "params_billion": params_billion,
                "download_gb": None,
                "downloads": max(0, int(model.get("downloads") or 0)),
            }

    # The search endpoint returns parameter counts but not blob sizes; details do.
    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = pool.map(_download_gb, candidates)
        for model, size in zip(candidates.values(), sizes):
            model["download_gb"] = size
    return list(candidates.values())


def _fallback_catalog() -> list[dict]:
    return [
        {
            "id": spec.model_id,
            "name": spec.display_name,
            "params_billion": spec.params_billion,
            "download_gb": None,
            "downloads": 0,
        }
        for spec in ModelRegistry().list_models()
    ]


def get_model_catalog() -> tuple[list[dict], str]:
    """Return cached Hub models, falling back to registered aliases on errors."""
    global _cached_at, _cached_catalog

    now = monotonic()
    if _cached_catalog is not None and now - _cached_at < _CACHE_SECONDS:
        models, source = _cached_catalog
        return [model.copy() for model in models], source

    try:
        models = _hub_catalog()
        if not models:
            raise ValueError("Hugging Face returned no models with parameter counts")
        source = "hub"
    except Exception:
        models, source = _fallback_catalog(), "fallback"

    _cached_at = monotonic()
    _cached_catalog = (models, source)
    return [model.copy() for model in models], source
