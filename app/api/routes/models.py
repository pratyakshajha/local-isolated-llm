"""Model discovery and active-model selection endpoints."""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_model_service
from app.services.model_service import ModelBusyError, ModelService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])


class ModelSelection(BaseModel):
    model_id: str = Field(min_length=1, max_length=256)


@router.get("/models")
def list_models(
    model_service: ModelService = Depends(get_model_service),
) -> dict[str, object]:
    return {
        "active_model": model_service.model_id,
        "loaded": model_service.is_loaded,
        "models": [
            {
                "id": model.model_id,
                "name": model.display_name,
                "runtime": model.runtime,
                "active": model.model_id == model_service.model_id,
            }
            for model in model_service.list_models()
        ],
    }


@router.post("/models/active")
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
