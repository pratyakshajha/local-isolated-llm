"""Small ASGI guards for request sizes and browser-facing security headers."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from starlette.datastructures import Headers

ASGIApp = Callable[[dict, Callable, Callable], Awaitable[None]]


def is_same_origin(origin: str, host: str) -> bool:
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc.casefold() == host.casefold()
    )


class RequestSizeLimitMiddleware:
    """Bound HTTP request buffering, including requests without Content-Length."""

    def __init__(self, app: ASGIApp, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        raw_length = headers.get("content-length")
        if raw_length is not None:
            try:
                if int(raw_length) > self.max_body_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send, status=400, detail="Invalid Content-Length")
                return

        messages: list[dict] = []
        body_size = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                break
            body_size += len(message.get("body", b""))
            if body_size > self.max_body_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break

        async def replay_receive() -> dict:
            if messages:
                return messages.pop(0)
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _reject(
        send: Callable, status: int = 413, detail: str = "Request body is too large"
    ) -> None:
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class BrowserSecurityMiddleware:
    """Reject cross-origin state changes and add safe defaults to HTTP replies."""

    _MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
    _SECURITY_HEADERS = [
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"no-referrer"),
    ]
    _CONTENT_SECURITY_POLICY = (
        b"content-security-policy",
        b"default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        b"script-src 'self'; connect-src 'self' ws: wss:; object-src 'none'; "
        b"base-uri 'self'; frame-ancestors 'none'",
    )

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            origin = headers.get("origin")
            host = headers.get("host", "")
            if (
                scope.get("method", "GET").upper() in self._MUTATING_METHODS
                and origin is not None
                and not is_same_origin(origin, host)
            ):
                await RequestSizeLimitMiddleware._reject(
                    send, status=403, detail="Cross-origin requests are not allowed"
                )
                return

            async def secure_send(message: dict) -> None:
                if message["type"] == "http.response.start":
                    message["headers"] = list(message.get("headers", [])) + self._SECURITY_HEADERS
                    path = scope.get("path", "")
                    if path not in {"/docs", "/redoc", "/openapi.json"}:
                        message["headers"].append(self._CONTENT_SECURITY_POLICY)
                await send(message)

            await self.app(scope, receive, secure_send)
            return

        await self.app(scope, receive, send)
