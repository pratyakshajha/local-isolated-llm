"""FastAPI application and server command."""

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

import uvicorn
from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.middleware import BrowserSecurityMiddleware, RequestSizeLimitMiddleware
from app.api.routes import health, models, openai, ui
from app.core.config import AppSettings, load_settings
from app.core.logging import setup_logging
from app.core.metrics import AppMetrics
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)


def get_public_base_url(settings: AppSettings, public_url: str | None = None) -> str:
    """Choose a browser-usable base URL, allowing proxy/container overrides."""
    configured_url = (
        os.environ.get("LOCAL_LLM_PUBLIC_URL", "") if public_url is None else public_url
    ).strip().rstrip("/")
    if configured_url:
        parsed = urlsplit(configured_url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return configured_url
        logger.warning(
            "Ignoring invalid LOCAL_LLM_PUBLIC_URL; expected an http(s) base URL"
        )

    host = settings.server.host
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    elif ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{settings.server.port}"


def create_app(
    settings: AppSettings | None = None,
    model_service: ModelService | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    model_service = model_service or ModelService(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        setup_logging()
        base_url = get_public_base_url(settings)
        logger.info(
            "Starting local LLM server on %s:%s",
            settings.server.host,
            settings.server.port,
        )
        if settings.server.host not in {"127.0.0.1", "::1", "localhost"}:
            logger.warning(
                "The server has no authentication and is listening beyond loopback; "
                "restrict network access at the firewall or reverse proxy"
            )
        if "*" in settings.server.allowed_hosts:
            logger.warning("Host validation is disabled by server.allowed_hosts='*'")
        logger.info(
            "Loading model %s. First startup may download model files; this can take several minutes.",
            model_service.model_name,
        )
        try:
            loaded_model = await asyncio.to_thread(model_service.load_model)
            logger.info("Model %s loaded", loaded_model)
            logger.info("Wonderdesk is ready!")
            logger.info("Web UI: %s/", base_url)
            logger.info("API documentation: %s/docs", base_url)
            logger.info("Readiness check: %s/health/ready", base_url)
            yield
        except Exception:
            logger.exception("Local LLM server failed during startup or runtime")
            raise
        finally:
            logger.info("Stopping local LLM server")
            await asyncio.to_thread(model_service.unload_model)
            logger.info("Local LLM server stopped")

    application = FastAPI(
        title="Local Isolated LLM",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.metrics = AppMetrics()
    application.state.settings = settings
    application.state.model_service = model_service
    allowed_hosts = set(settings.server.allowed_hosts)
    if settings.server.host not in {"0.0.0.0", "::", ""}:
        allowed_hosts.add(settings.server.host)
    application.add_middleware(
        TrustedHostMiddleware, allowed_hosts=sorted(allowed_hosts)
    )
    application.add_middleware(
        RequestSizeLimitMiddleware,
        max_body_bytes=settings.server.max_request_bytes,
    )
    application.add_middleware(BrowserSecurityMiddleware)

    @application.middleware("http")
    async def observe_requests(request, call_next):
        started_at = perf_counter()
        request_id = uuid4().hex[:12]
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            duration = perf_counter() - started_at
            route = request.scope.get("route")
            route_path = getattr(route, "path", "<unmatched>")
            application.state.metrics.observe_http(
                request.method, route_path, status, duration
            )
            logger.exception(
                "request_id=%s method=%s route=%s status=%d duration_ms=%.2f",
                request_id,
                request.method,
                route_path,
                status,
                duration * 1000,
            )
            raise
        duration = perf_counter() - started_at
        route = request.scope.get("route")
        route_path = getattr(route, "path", "<unmatched>")
        application.state.metrics.observe_http(
            request.method, route_path, status, duration
        )
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s method=%s route=%s status=%d duration_ms=%.2f",
            request_id,
            request.method,
            route_path,
            status,
            duration * 1000,
        )
        return response

    application.include_router(health.router)
    application.include_router(models.router)
    application.include_router(openai.router)
    application.include_router(ui.router)
    return application


app = create_app()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the local LLM API server")
    parser.add_argument("--config", help="Path to a YAML config file")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    setup_logging()
    uvicorn.run(
        create_app(settings),
        host=settings.server.host,
        port=settings.server.port,
        log_config=None,
        ws_max_size=settings.server.max_request_bytes,
        ws_max_queue=4,
        limit_concurrency=16,
        timeout_keep_alive=5,
        timeout_graceful_shutdown=30,
        server_header=False,
    )


if __name__ == "__main__":
    main()
