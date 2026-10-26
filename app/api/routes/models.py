"""Model discovery and active-model selection endpoints."""

import asyncio
import logging
import math

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.dependencies import get_model_service
from app.services.hardware_fit import analyze_models, detect_hardware
from app.services.model_cache import get_downloaded_model_ids
from app.services.model_catalog import get_model_catalog
from app.services.model_service import ModelBusyError, ModelService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])


@router.get("/api/model-fit")
async def model_fit(
    gpu_vram_gb: float | None = Query(default=None, ge=0),
    ram_gb: float | None = Query(default=None, ge=0),
) -> dict[str, object]:
    for name, value in (("gpu_vram_gb", gpu_vram_gb), ("ram_gb", ram_gb)):
        if value is not None and not math.isfinite(value):
            raise HTTPException(status_code=422, detail=f"{name} must be finite")

    detected = detect_hardware()
    used = {
        "gpu_vram_gb": detected["gpu_vram_gb"] if gpu_vram_gb is None else gpu_vram_gb,
        "ram_gb": detected["ram_gb"] if ram_gb is None else ram_gb,
    }
    backend = detected["backend"]
    if gpu_vram_gb is not None:
        backend = "cuda" if gpu_vram_gb > 0 else "cpu"

    catalog, catalog_source = await asyncio.to_thread(get_model_catalog)
    analysis = analyze_models(
        catalog,
        gpu_vram_gb=used["gpu_vram_gb"],
        ram_gb=used["ram_gb"],
        disk_free_gb=detected["disk_free_gb"],
        gpu_name=detected["gpu_name"],
        backend=backend,
    )
    downloaded_model_ids = await asyncio.to_thread(
        get_downloaded_model_ids,
        {model["id"] for model in analysis["models"]},
    )
    public_model_fields = (
        "id",
        "name",
        "params_billion",
        "download_gb",
        "need_gb",
        "verdict",
        "disk_ok",
        "est_tokens_per_sec",
    )
    return {
        "detected": detected,
        "used": used,
        "catalog_source": catalog_source,
        "best_fit": analysis["best_fit"],
        "models": [
            {
                **{key: model[key] for key in public_model_fields},
                "downloaded": model["id"] in downloaded_model_ids,
            }
            for model in analysis["models"]
        ],
    }


class ModelSelection(BaseModel):
    model_id: str = Field(min_length=1, max_length=256)


@router.get("/api/models/cached")
async def list_cached_models(
    model_service: ModelService = Depends(get_model_service),
) -> dict[str, object]:
    cached_ids = await asyncio.to_thread(get_downloaded_model_ids)
    specs = [model_service.model_registry.resolve(model_id) for model_id in sorted(cached_ids)]
    return {
        "active_model": model_service.model_id,
        "models": [
            {
                "id": model.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "name": model.display_name,
                "runtime": model.runtime,
                "supports_chat": model.supports_chat,
                "active": model.model_id == model_service.model_id,
            }
            for model in specs
        ],
    }


@router.get("/v1/models")
def list_openai_models(
    model_service: ModelService = Depends(get_model_service),
) -> dict[str, object]:
    active_model = model_service.model_registry.resolve(model_service.model_id)
    specs = {model.model_id: model for model in model_service.list_models()}
    specs.setdefault(active_model.model_id, active_model)
    return {
        "object": "list",
        "active_model": model_service.model_id,
        "loaded": model_service.is_loaded,
        "data": [
            {
                "id": model.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "name": model.display_name,
                "runtime": model.runtime,
                "supports_chat": model.supports_chat,
                "active": model.model_id == model_service.model_id,
            }
            for model in specs.values()
        ],
    }


@router.post("/v1/models/active")
async def select_model(
    selection: ModelSelection,
    model_service: ModelService = Depends(get_model_service),
) -> dict[str, str]:
    try:
        model_id = await asyncio.to_thread(
            model_service.switch_model, selection.model_id
        )
    except ModelBusyError as exc:
        raise HTTPException(
            status_code=429, detail="The model is busy; try again shortly"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unable to switch model to %s", selection.model_id)
        raise HTTPException(status_code=500, detail="Unable to load selected model") from exc
    return {"active_model": model_id}
