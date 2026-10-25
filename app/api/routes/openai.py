"""OpenAI-compatible chat completion endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
from threading import Event, Thread
from time import perf_counter, time
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from app.api.dependencies import get_model_service
from app.core.config import GenerationSettings
from app.llm.tools import inject_tools_into_messages, parse_tool_calls
from app.services.model_service import ModelBusyError, ModelService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["OpenAI-compatible"])


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class FunctionCallInput(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    arguments: str

    @field_validator("arguments")
    @classmethod
    def arguments_must_be_an_object(cls, value: str) -> str:
        try:
            arguments = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("tool call arguments must be a JSON object") from exc
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must be a JSON object")
        return value


class ToolCallInput(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    type: Literal["function"]
    function: FunctionCallInput


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[TextPart] | None = None
    tool_calls: list[ToolCallInput] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_tool_message(self) -> ChatMessage:
        if self.tool_calls and self.role != "assistant":
            raise ValueError("tool_calls are only valid on assistant messages")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        return self


class FunctionDefinition(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class FunctionTool(BaseModel):
    type: Literal["function"]
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    top_p: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    max_tokens: int | None = Field(default=None, ge=1, le=32_768)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32_768)
    stop: str | list[str] | None = None
    tools: list[FunctionTool] | None = Field(default=None, max_length=128)
    tool_choice: str | dict[str, Any] | None = None


def _normalize_messages(
    messages: list[ChatMessage], max_message_chars: int
) -> list[dict[str, Any]]:
    normalized = []
    total_chars = 0
    for message in messages:
        content = message.content if isinstance(message.content, str) else ""
        if isinstance(message.content, list):
            content = "".join(part.text for part in message.content)
        total_chars += len(content)
        if total_chars > max_message_chars:
            raise HTTPException(
                status_code=413, detail="Messages exceed the configured size limit"
            )
        normalized_message: dict[str, Any] = {
            "role": "system" if message.role == "developer" else message.role,
            "content": content,
        }
        if message.tool_calls:
            normalized_message["tool_calls"] = [
                call.model_dump() for call in message.tool_calls
            ]
        if message.tool_call_id:
            normalized_message["tool_call_id"] = message.tool_call_id
        normalized.append(normalized_message)
    return normalized


def _coalesce_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_messages = [message for message in messages if message["role"] == "system"]
    normalized = []
    if system_messages:
        normalized.append(
            {
                "role": "system",
                "content": "\n\n".join(
                    message["content"]
                    for message in system_messages
                    if message["content"]
                ),
            }
        )

    for message in messages:
        if message["role"] == "system":
            continue
        if normalized and normalized[-1]["role"] == message["role"]:
            previous = normalized[-1]["content"]
            current = message["content"]
            normalized[-1]["content"] = "\n\n".join(
                content for content in (previous, current) if content
            )
        else:
            normalized.append(message)
    return normalized


def _validated_tools(
    payload: ChatCompletionRequest,
) -> tuple[list[dict[str, Any]], str | dict[str, Any] | None]:
    tools = [tool.model_dump(exclude_none=True) for tool in payload.tools or []]
    names = [tool["function"]["name"] for tool in tools]
    if len(names) != len(set(names)):
        raise HTTPException(
            status_code=422, detail="Tool function names must be unique"
        )
    choice = payload.tool_choice
    if choice is not None and not isinstance(choice, (str, dict)):
        raise HTTPException(status_code=422, detail="Invalid tool_choice")
    if isinstance(choice, str) and choice not in {"auto", "none", "required"}:
        raise HTTPException(status_code=422, detail="Invalid tool_choice")
    if isinstance(choice, dict):
        function_choice = choice.get("function")
        if choice.get("type") != "function" or not isinstance(function_choice, dict):
            raise HTTPException(status_code=422, detail="Invalid tool_choice")
        name = function_choice.get("name")
        if name not in names:
            raise HTTPException(
                status_code=422, detail="tool_choice must name a requested function"
            )
    if choice is not None and not tools and choice != "none":
        raise HTTPException(status_code=422, detail="tool_choice requires tools")
    return tools, choice


def _tool_response(
    text: str,
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] | None,
) -> tuple[str | None, list[dict[str, Any]]]:
    if not tools or tool_choice == "none":
        return text, []
    content, calls = parse_tool_calls(text)
    allowed_names = {tool["function"]["name"] for tool in tools}
    if isinstance(tool_choice, dict):
        allowed_names = {tool_choice["function"]["name"]}
    allowed = [call for call in calls if call["function"]["name"] in allowed_names]
    rejected = [call for call in calls if call["function"]["name"] not in allowed_names]
    if rejected:
        rejected_text = "".join(
            "<tool_call>"
            + json.dumps(
                {
                    "name": call["function"]["name"],
                    "arguments": json.loads(call["function"]["arguments"]),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "</tool_call>"
            for call in rejected
        )
        content = f"{content or ''}{rejected_text}"
    return content, allowed


def _generation_settings(
    payload: ChatCompletionRequest, defaults: GenerationSettings
) -> GenerationSettings:
    max_tokens = payload.max_completion_tokens or payload.max_tokens
    return GenerationSettings(
        temperature=(
            defaults.temperature if payload.temperature is None else payload.temperature
        ),
        top_p=defaults.top_p if payload.top_p is None else payload.top_p,
        max_new_tokens=defaults.max_new_tokens if max_tokens is None else max_tokens,
    )


def _stops(payload: ChatCompletionRequest) -> list[str]:
    stops = [payload.stop] if isinstance(payload.stop, str) else payload.stop or []
    return [stop for stop in stops if stop]


def _truncate_at_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    matches = [(text.find(stop), stop) for stop in stops if stop in text]
    if not matches:
        return text, False
    position, _ = min(matches)
    return text[:position], True


def _completion_id() -> str:
    return f"chatcmpl-{uuid4().hex}"


def _sse(data: dict[str, object] | str) -> str:
    if isinstance(data, str):
        return f"data: {data}\n\n"
    return f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, str],
    finish_reason: str | None = None,
) -> dict[str, object]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def _select_model(model_service: ModelService, requested: str | None) -> str:
    if requested is None:
        return model_service.model_id
    try:
        resolved = model_service.model_registry.resolve(requested)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if resolved.model_id != model_service.model_id:
        try:
            await asyncio.to_thread(model_service.switch_model, resolved.model_id)
        except ModelBusyError as exc:
            raise HTTPException(
                status_code=429, detail="The model is busy; try again shortly"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return model_service.model_id


@router.post("/chat/completions")
async def create_chat_completion(
    payload: ChatCompletionRequest,
    request: Request,
    model_service: ModelService = Depends(get_model_service),
):
    messages = _normalize_messages(
        payload.messages, request.app.state.settings.server.max_message_chars
    )
    tools, tool_choice = _validated_tools(payload)
    if tools or any(
        message.get("tool_calls") or message["role"] == "tool" for message in messages
    ):
        messages = inject_tools_into_messages(messages, tools, tool_choice)
    messages = _coalesce_messages(messages)
    if (
        sum(len(message["content"]) for message in messages)
        > request.app.state.settings.server.max_message_chars
    ):
        raise HTTPException(
            status_code=413, detail="Messages exceed the configured size limit"
        )
    model = await _select_model(model_service, payload.model)
    generation = _generation_settings(payload, request.app.state.settings.generation)
    stops = _stops(payload)
    completion_id = _completion_id()
    created = int(time())

    if not payload.stream:
        started = perf_counter()
        try:
            content = await asyncio.to_thread(
                model_service.generate, messages, generation
            )
            content, _ = _truncate_at_stop(content, stops)
            content, tool_calls = _tool_response(content, tools, tool_choice)
        except ModelBusyError as exc:
            request.app.state.metrics.observe_generation(
                "busy", perf_counter() - started
            )
            raise HTTPException(
                status_code=429, detail="The model is busy; try again shortly"
            ) from exc
        except Exception as exc:
            request.app.state.metrics.observe_generation(
                "failed", perf_counter() - started
            )
            logger.exception("OpenAI-compatible chat completion failed")
            raise HTTPException(
                status_code=500, detail="Chat completion failed"
            ) from exc
        request.app.state.metrics.observe_generation(
            "success", perf_counter() - started
        )
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
        }

    cancel_event = Event()
    stream_queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def publish(kind: str, value: object) -> None:
        try:
            loop.call_soon_threadsafe(stream_queue.put_nowait, (kind, value))
        except RuntimeError:
            cancel_event.set()

    def produce() -> None:
        started = perf_counter()
        outcome = "success"
        try:
            for text in model_service.generate_stream(
                messages, generation, cancel_event=cancel_event
            ):
                publish("text", text)
        except Exception as exc:
            outcome = "failed"
            publish("error", exc)
        finally:
            if cancel_event.is_set():
                outcome = "cancelled"
            request.app.state.metrics.observe_generation(
                outcome, perf_counter() - started
            )
            publish("done", None)

    worker = Thread(target=produce, name="openai-chat-stream", daemon=True)
    worker.start()

    async def events():
        finish_reason = "stop"
        pending = ""
        tool_output: list[str] = []
        tool_mode = bool(tools and tool_choice != "none")
        max_stop_length = max((len(stop) for stop in stops), default=0)
        try:
            yield _sse(_chunk(completion_id, created, model, {"role": "assistant"}))
            while True:
                kind, value = await stream_queue.get()
                if kind == "text":
                    if tool_mode:
                        tool_output.append(str(value))
                        continue
                    pending += str(value)
                    found = [
                        (pending.find(stop), stop) for stop in stops if stop in pending
                    ]
                    if found:
                        position, _ = min(found)
                        if pending[:position]:
                            yield _sse(
                                _chunk(
                                    completion_id,
                                    created,
                                    model,
                                    {"content": pending[:position]},
                                )
                            )
                        pending = ""
                        finish_reason = "stop"
                        cancel_event.set()
                        continue
                    safe_length = len(pending) - max_stop_length + 1
                    if safe_length > 0:
                        yield _sse(
                            _chunk(
                                completion_id,
                                created,
                                model,
                                {"content": pending[:safe_length]},
                            )
                        )
                        pending = pending[safe_length:]
                elif kind == "error":
                    logger.error("Streaming chat completion failed: %s", value)
                    finish_reason = "error"
                elif kind == "done":
                    if tool_mode:
                        raw_output, _ = _truncate_at_stop("".join(tool_output), stops)
                        content, tool_calls = _tool_response(
                            raw_output, tools, tool_choice
                        )
                        if tool_calls:
                            if content:
                                yield _sse(
                                    _chunk(
                                        completion_id,
                                        created,
                                        model,
                                        {"content": content},
                                    )
                                )
                            for index, call in enumerate(tool_calls):
                                yield _sse(
                                    _chunk(
                                        completion_id,
                                        created,
                                        model,
                                        {
                                            "tool_calls": [
                                                {
                                                    "index": index,
                                                    "id": call["id"],
                                                    "type": "function",
                                                    "function": call["function"],
                                                }
                                            ]
                                        },
                                    )
                                )
                            finish_reason = "tool_calls"
                        elif content:
                            yield _sse(
                                _chunk(
                                    completion_id,
                                    created,
                                    model,
                                    {"content": content},
                                )
                            )
                        break
                    if pending and not cancel_event.is_set():
                        yield _sse(
                            _chunk(
                                completion_id,
                                created,
                                model,
                                {"content": pending},
                            )
                        )
                    break
            yield _sse(_chunk(completion_id, created, model, {}, finish_reason))
            yield _sse("[DONE]")
        finally:
            cancel_event.set()
            if worker.is_alive():
                await asyncio.to_thread(worker.join, 1)

    return StreamingResponse(events(), media_type="text/event-stream")
