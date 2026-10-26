"""Normalize OpenAI tool messages for local causal language models."""

from __future__ import annotations

import html
import json
import re
from typing import Any, Sequence
from uuid import uuid4

_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _make_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    arguments = value.get("arguments")
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None
    return {
        "id": f"call_{uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": _json(arguments),
        },
    }


def _tool_instruction(tools: Sequence[dict[str, Any]], tool_choice: Any) -> str:
    instruction = (
        "Use only the listed tools. When needed, emit "
        '<tool_call>{"name":"FUNCTION_NAME","arguments":{}}</tool_call> '
        "with the real function name and arguments. Do not describe a tool call "
        "in prose. Otherwise answer normally."
    )
    if tool_choice == "required":
        # shortcut: required choices are prompt-enforced only, add constrained decoding when a runtime supports it.
        instruction += " You must call a function."
    elif isinstance(tool_choice, dict):
        name = tool_choice.get("function", {}).get("name")
        instruction += f" You must call the function {name!r}."
    return f"{instruction}\n<tools>{_json(tools)}</tools>"


def inject_tools_into_messages(
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Convert OpenAI tool history to plain chat-template messages."""
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = call.get("function", {})
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError(
                        "tool call history arguments must be a JSON object"
                    )
                calls.append(
                    {
                        "name": function.get("name", ""),
                        "arguments": arguments,
                    }
                )
            call_text = "".join(
                "<tool_call>"
                + _json(call).replace("<", "\\u003c").replace(">", "\\u003e")
                + "</tool_call>"
                for call in calls
            )
            text = f"{text}\n{call_text}".strip()
        elif role == "tool":
            call_id = html.escape(str(message.get("tool_call_id", "")), quote=True)
            text = f'<tool_response id="{call_id}">{html.escape(text)}</tool_response>'
            role = "user"
        normalized.append({"role": role, "content": text})

    if tools and tool_choice != "none":
        instruction = _tool_instruction(tools, tool_choice)
        system = next(
            (message for message in normalized if message["role"] == "system"), None
        )
        if system is None:
            normalized.insert(0, {"role": "system", "content": instruction})
        else:
            system["content"] = f"{system['content']}\n\n{instruction}".strip()
    return normalized


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract valid object-argument tool calls, preserving malformed text."""
    stripped = text.strip()
    try:
        bare = _make_call(json.loads(stripped))
    except json.JSONDecodeError:
        bare = None
    if bare is not None:
        return None, [bare]

    calls: list[dict[str, Any]] = []
    remaining: list[str] = []
    last = 0
    for match in _TOOL_CALL.finditer(text):
        try:
            call = _make_call(json.loads(match.group(1)))
        except json.JSONDecodeError:
            call = None
        if call is None:
            continue
        remaining.append(text[last : match.start()])
        calls.append(call)
        last = match.end()
    if not calls:
        return text, []
    remaining.append(text[last:])
    content = "".join(remaining).strip()
    return content or None, calls
