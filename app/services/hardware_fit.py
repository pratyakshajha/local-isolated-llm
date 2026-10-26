"""Memory and rough throughput estimates for local model candidates."""

import math
import os
import shutil
from pathlib import Path

_GIB = 1024**3
_GPU_BANDWIDTH_GBPS = {
    "rtx 5060 laptop": 384,
    "rtx 4090": 1008,
    "rtx 4080": 716.8,
    "rtx 4070": 504,
    "rtx 3090": 936,
    "rx 7900 xtx": 960,
    "a100": 1555,
    "h100": 3350,
    "m3 max": 400,
    "m4 max": 546,
}


def estimate_bytes(params_billion: float | None, bytes_per_param: int) -> int | None:
    if params_billion is None:
        return None
    if not math.isfinite(params_billion) or params_billion <= 0:
        raise ValueError("params_billion must be finite and positive")
    if bytes_per_param not in (2, 4):
        raise ValueError("bytes_per_param must be 2 or 4")
    # shortcut: fixed 20% plus 1 GiB margin, upgrade when users report OOMs.
    return math.ceil(params_billion * 1_000_000_000 * bytes_per_param * 1.2 + _GIB)


def _gpu_bandwidth_gbps(gpu_name: str | None) -> float | None:
    if not gpu_name:
        return None
    name = gpu_name.casefold()
    return next(
        (bandwidth for model, bandwidth in _GPU_BANDWIDTH_GBPS.items() if model in name),
        None,
    )


def detect_hardware() -> dict[str, object]:
    """Detect host memory, accelerator backend, and free Hugging Face cache space."""
    notes = []
    try:
        import torch
    except (ImportError, OSError):
        torch = None
        notes.append("PyTorch is unavailable; GPU detection was skipped.")

    import psutil

    ram_bytes = psutil.virtual_memory().total
    backend = "cpu"
    gpu_name = None
    gpu_bytes = 0

    if torch is not None:
        if torch.cuda.is_available():
            backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            properties = torch.cuda.get_device_properties(0)
            gpu_name = getattr(properties, "name", None)
            gpu_bytes = getattr(properties, "total_memory", 0)
        elif getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available():
            backend = "mps"
            gpu_bytes = ram_bytes
            notes.append("MPS uses unified memory; fit estimates follow the CPU runtime path.")
        elif getattr(torch, "xpu", None) and torch.xpu.is_available():
            backend = "xpu"
            try:
                properties = torch.xpu.get_device_properties(0)
                gpu_name = getattr(properties, "name", None)
                gpu_bytes = getattr(properties, "total_memory", 0)
            except (AttributeError, RuntimeError):
                notes.append("XPU is available, but its name and memory could not be read.")
            notes.append("XPU fit estimates follow the CPU runtime path.")
        else:
            cuda_version = getattr(torch.version, "cuda", None)
            if cuda_version is None:
                notes.append(
                    "PyTorch was installed without CUDA support; "
                    "GPU detection requires a CUDA/ROCm build of PyTorch."
                )
            else:
                notes.append("CUDA is available but no compatible GPU was found.")

    cache_home = Path(
        os.environ.get("HF_HOME", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface")
    )
    cache_path = Path(os.environ.get("HF_HUB_CACHE", cache_home / "hub")).expanduser()
    while not cache_path.exists() and cache_path != cache_path.parent:
        cache_path = cache_path.parent
    disk_bytes = shutil.disk_usage(cache_path).free

    return {
        "backend": backend,
        "gpu_name": gpu_name,
        "gpu_vram_gb": round(gpu_bytes / 1_000_000_000, 2),
        "ram_gb": round(ram_bytes / 1_000_000_000, 2),
        "disk_free_gb": round(disk_bytes / 1_000_000_000, 2),
        "notes": notes,
    }


def analyze_models(
    models: list[dict],
    *,
    gpu_vram_gb: float,
    ram_gb: float,
    disk_free_gb: float,
    gpu_name: str | None,
    backend: str = "cuda",
) -> dict[str, object]:
    """Estimate each model's fit and select the largest one that fits."""
    use_gpu = gpu_vram_gb > 0 and backend not in ("mps", "xpu")
    bytes_per_param = 2 if use_gpu else 4
    available_bytes = (gpu_vram_gb if use_gpu else ram_gb) * 1_000_000_000
    bandwidth = _gpu_bandwidth_gbps(gpu_name) if use_gpu else None
    results = []

    for model in models:
        params_billion = model["params_billion"]
        need_bytes = estimate_bytes(params_billion, bytes_per_param)
        download_gb = model.get("download_gb")
        disk_ok = None if download_gb is None else disk_free_gb >= download_gb
        if params_billion is None:
            verdict = "unknown_size"
        elif need_bytes <= available_bytes:
            verdict = "fits_gpu" if use_gpu else "fits_cpu"
        else:
            verdict = "wont_fit"

        results.append(
            {
                "id": model["id"],
                "name": model["name"],
                "params_billion": params_billion,
                "download_gb": download_gb,
                "downloads": model.get("downloads", 0),
                "need_gb": None if need_bytes is None else round(need_bytes / 1_000_000_000, 2),
                "verdict": verdict,
                "disk_ok": disk_ok,
                "est_tokens_per_sec": (
                    None
                    if bandwidth is None or params_billion is None
                    else round(bandwidth / (params_billion * 2), 1)
                ),
            }
        )

    candidates = [
        model
        for model in results
        if model["verdict"] in ("fits_gpu", "fits_cpu") and model["disk_ok"] is not False
    ]
    best_fit = max(
        candidates,
        key=lambda model: (model["params_billion"], model.get("downloads", 0)),
        default=None,
    )
    return {"models": results, "best_fit": None if best_fit is None else best_fit["id"]}
