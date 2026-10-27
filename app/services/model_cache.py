"""Report model repositories with complete usable snapshots in the HF cache."""

import json

from huggingface_hub import scan_cache_dir
from huggingface_hub.utils import CacheNotFound

_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.txt",
}
_WEIGHT_FILES = {"model.safetensors", "model.bin", "pytorch_model.bin"}
_WEIGHT_INDEX_SUFFIXES = {
    "model.safetensors.index.json": ".safetensors",
    "model.bin.index.json": ".bin",
    "pytorch_model.bin.index.json": ".bin",
}


def _snapshot_is_complete(snapshot) -> bool:
    files = {file.file_name: file for file in snapshot.files}
    has_tokenizer = bool(_TOKENIZER_FILES.intersection(files)) or {
        "vocab.json",
        "merges.txt",
    }.issubset(files)
    if "config.json" not in files or not has_tokenizer:
        return False
    if _WEIGHT_FILES.intersection(files):
        return True

    for index_name, weight_suffix in _WEIGHT_INDEX_SUFFIXES.items():
        if index_name not in files:
            continue
        try:
            index = json.loads(files[index_name].file_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if (
            isinstance(weight_map, dict)
            and weight_map
            and all(
                isinstance(shard, str)
                and shard.endswith(weight_suffix)
                and shard in files
                for shard in weight_map.values()
            )
        ):
            return True
    return False


def get_downloaded_model_ids(model_ids: set[str] | None = None) -> set[str]:
    """Return IDs with config, tokenizer, and complete weights cached locally."""
    if model_ids is not None and not model_ids:
        return set()
    try:
        cache = scan_cache_dir()
    except (CacheNotFound, OSError, ValueError):
        return set()

    return {
        repo.repo_id
        for repo in cache.repos
        if repo.repo_type == "model"
        and (model_ids is None or repo.repo_id in model_ids)
        and any(_snapshot_is_complete(snapshot) for snapshot in repo.revisions)
    }
