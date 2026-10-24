"""REST and WebSocket chat endpoints."""

import asyncio
import json
import logging
from threading import Event
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.api.dependencies import get_chat_service
from app.api.middleware import is_same_origin
from app.services.chat_service import ChatService, SessionLimitError
from app.services.model_service import ModelBusyError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=16_384)
    session_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    session_id: str
    model: str
    response: str


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    http_request: Request,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    max_message_chars = http_request.app.state.settings.server.max_message_chars
    if len(payload.message) > max_message_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Message exceeds the {max_message_chars}-character limit",
        )
    created_session = payload.session_id is None
    try:
        session_id = payload.session_id or chat_service.start_session()
    except SessionLimitError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    started_at = perf_counter()
    try:
        response = await asyncio.to_thread(
            chat_service.generate, session_id, payload.message
        )
    except ModelBusyError as exc:
        http_request.app.state.metrics.observe_generation(
            "busy", perf_counter() - started_at
        )
        if created_session:
            chat_service.close_session(session_id)
        raise HTTPException(status_code=429, detail="The model is busy; try again shortly") from exc
    except KeyError as exc:
        http_request.app.state.metrics.observe_generation(
            "error", perf_counter() - started_at
        )
        if created_session:
            chat_service.close_session(session_id)
        raise HTTPException(status_code=404, detail="Chat session not found") from exc
    except ValueError as exc:
        http_request.app.state.metrics.observe_generation(
            "error", perf_counter() - started_at
        )
        if created_session:
            chat_service.close_session(session_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        http_request.app.state.metrics.observe_generation(
            "error", perf_counter() - started_at
        )
        if created_session:
            chat_service.close_session(session_id)
        logger.exception("Chat generation failed")
        raise HTTPException(
            status_code=500, detail="Chat response generation failed"
        ) from exc
    http_request.app.state.metrics.observe_generation(
        "success", perf_counter() - started_at
    )
    return ChatResponse(
        session_id=session_id,
        model=chat_service.model_name,
        response=response,
    )


@router.websocket("/")
@router.websocket("/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    chat_service: ChatService = Depends(get_chat_service),
) -> None:
    origin = websocket.headers.get("origin")
    host = websocket.headers.get("host", "")
    if origin and not is_same_origin(origin, host):
        await websocket.close(code=1008, reason="Origin is not allowed")
        return
    await websocket.accept()
    try:
        session_id = chat_service.start_session(retain=True)
    except SessionLimitError:
        if websocket.query_params.get("protocol") == "json":
            await websocket.send_json(
                {"type": "error", "content": "The chat session limit has been reached."}
            )
        await websocket.close(code=1013, reason="Chat session limit reached")
        return
    use_json_protocol = websocket.query_params.get("protocol") == "json"
    generation_task: asyncio.Task[str | None] | None = None
    generation_cancel_event: Event | None = None
    generation_was_cancelled = False
    generation_started_at: float | None = None
    receive_task = asyncio.create_task(websocket.receive_text())

    def record_generation(outcome: str, duration: float) -> None:
        websocket.app.state.metrics.observe_generation(outcome, duration)
        logger.info(
            "session_id=%s generation_outcome=%s duration_ms=%.2f",
            session_id,
            outcome,
            duration * 1000,
        )

    try:
        logger.info(
            "session_id=%s websocket_connected protocol=%s",
            session_id,
            "json" if use_json_protocol else "legacy",
        )
        if use_json_protocol:
            await websocket.send_json(
                {"type": "ready", "model": chat_service.model_name}
            )
        else:
            await websocket.send_text(f"{chat_service.model_name} is ready to chat!")
        while True:
            pending_tasks: set[asyncio.Task] = {receive_task}
            if generation_task is not None:
                pending_tasks.add(generation_task)
            done, _ = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            if generation_task is not None and generation_task in done:
                completed_task = generation_task
                duration = (
                    perf_counter() - generation_started_at
                    if generation_started_at is not None
                    else 0.0
                )
                was_cancelled = generation_was_cancelled or bool(
                    generation_cancel_event and generation_cancel_event.is_set()
                )
                generation_task = None
                generation_cancel_event = None
                generation_was_cancelled = False
                generation_started_at = None
                try:
                    response = completed_task.result()
                except ValueError as exc:
                    record_generation("error", duration)
                    if use_json_protocol:
                        await websocket.send_json({"type": "error", "content": str(exc)})
                    else:
                        await websocket.send_text(f"Error: {exc}")
                except ModelBusyError:
                    record_generation("busy", duration)
                    if use_json_protocol:
                        await websocket.send_json(
                            {"type": "error", "content": "The model is busy; try again shortly."}
                        )
                    else:
                        await websocket.send_text("Error: the model is busy; try again shortly")
                except Exception:
                    record_generation("error", duration)
                    logger.exception(
                        "session_id=%s WebSocket chat generation failed", session_id
                    )
                    if use_json_protocol:
                        await websocket.send_json(
                            {"type": "error", "content": "Response generation failed."}
                        )
                    else:
                        await websocket.send_text("Error: response generation failed")
                else:
                    if was_cancelled or response is None:
                        record_generation("cancelled", duration)
                        if use_json_protocol:
                            await websocket.send_json({"type": "generation_cancelled"})
                        else:
                            await websocket.send_text("Generation cancelled.")
                    elif use_json_protocol:
                        record_generation("success", duration)
                        await websocket.send_json(
                            {
                                "type": "assistant_message",
                                "model": chat_service.model_name,
                                "content": response,
                            }
                        )
                    else:
                        record_generation("success", duration)
                        await websocket.send_text(f"{chat_service.model_name}: {response}")
                continue

            raw_message = receive_task.result()
            receive_task = asyncio.create_task(websocket.receive_text())
            if use_json_protocol:
                try:
                    event = json.loads(raw_message)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "content": "Message must be valid JSON."}
                    )
                    continue
                if isinstance(event, dict) and event.get("type") == "cancel_generation":
                    if generation_cancel_event is not None:
                        generation_was_cancelled = True
                        generation_cancel_event.set()
                    continue
                if (
                    not isinstance(event, dict)
                    or event.get("type") != "user_message"
                    or not isinstance(event.get("content"), str)
                ):
                    await websocket.send_json(
                        {"type": "error", "content": "Expected a user_message event."}
                    )
                    continue
                message = event["content"]
            else:
                message = raw_message
            max_message_chars = websocket.app.state.settings.server.max_message_chars
            if len(message) > max_message_chars:
                if use_json_protocol:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "content": f"Message exceeds the {max_message_chars}-character limit",
                        }
                    )
                else:
                    await websocket.send_text(
                        f"Error: message exceeds the {max_message_chars}-character limit"
                    )
                continue
            if generation_task is not None:
                if use_json_protocol:
                    await websocket.send_json(
                        {"type": "error", "content": "A response is already being generated."}
                    )
                else:
                    await websocket.send_text("Error: a response is already being generated")
                continue

            generation_cancel_event = Event()
            generation_was_cancelled = False
            generation_started_at = perf_counter()
            generation_task = asyncio.create_task(
                asyncio.to_thread(
                    chat_service.generate,
                    session_id,
                    message,
                    generation_cancel_event,
                )
            )
    except WebSocketDisconnect:
        logger.info("session_id=%s websocket_disconnected", session_id)
    finally:
        receive_task.cancel()
        await asyncio.gather(receive_task, return_exceptions=True)
        if generation_cancel_event is not None:
            generation_cancel_event.set()
        if generation_task is not None:
            duration = (
                perf_counter() - generation_started_at
                if generation_started_at is not None
                else 0.0
            )
            try:
                await generation_task
                record_generation("disconnected", duration)
            except Exception:
                logger.debug(
                    "session_id=%s generation ended while closing the WebSocket",
                    session_id,
                )
        chat_service.close_session(session_id)
