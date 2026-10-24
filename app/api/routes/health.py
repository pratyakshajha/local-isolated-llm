"""Health endpoint."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    model_service = request.app.state.model_service
    return {
        "status": "ok" if model_service.is_loaded else "not_ready",
        "model_loaded": model_service.is_loaded,
    }


@router.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
def readiness(request: Request) -> dict[str, object]:
    model_service = request.app.state.model_service
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="The model is not ready")
    return {"status": "ready", "model": model_service.model_id}


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def metrics(request: Request) -> PlainTextResponse:
    model_service = request.app.state.model_service
    chat_service = request.app.state.chat_service
    try:
        import psutil

        resident_memory = psutil.Process().memory_info().rss
    except ImportError:
        try:
            import resource

            resident_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if resident_memory < 10_000_000:  # Linux reports KiB; macOS reports bytes.
                resident_memory *= 1024
        except ImportError:  # pragma: no cover - psutil is a runtime dependency on Windows.
            resident_memory = 0
    body = request.app.state.metrics.render(
        model_loaded=model_service.is_loaded,
        active_sessions=chat_service.active_sessions,
        resident_memory=resident_memory,
    )
    return PlainTextResponse(
        body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )
