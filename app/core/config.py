"""Validated application configuration loaded from YAML."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "user_config.yaml"
DEFAULT_MODEL_ID = "google/gemma-3-1b-it"


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 2048

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("generation.temperature must be finite and non-negative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("generation.top_p must be greater than 0 and at most 1")
        if self.max_new_tokens < 1:
            raise ValueError("generation.max_tokens must be at least 1")
        if self.max_new_tokens > 32_768:
            raise ValueError("generation.max_new_tokens must not exceed 32768")


@dataclass(frozen=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8080
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")
    max_request_bytes: int = 4_194_304
    max_message_chars: int = 262_144
    max_queued_requests: int = 4
    queue_timeout_seconds: float = 30.0

    def __post_init__(self):
        if not self.host:
            raise ValueError("server.host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("server.port must be between 1 and 65535")
        if not self.allowed_hosts or any(not host.strip() for host in self.allowed_hosts):
            raise ValueError("server.allowed_hosts must contain non-empty host names")
        if not 1 <= self.max_request_bytes <= 16_777_216:
            raise ValueError(
                "server.max_request_bytes must be between 1 and 16777216"
            )
        if not 1 <= self.max_message_chars <= 1_048_576:
            raise ValueError(
                "server.max_message_chars must be between 1 and 1048576"
            )
        if not 0 <= self.max_queued_requests <= 64:
            raise ValueError("server.max_queued_requests must be between 0 and 64")
        if (
            not math.isfinite(self.queue_timeout_seconds)
            or not 0 <= self.queue_timeout_seconds <= 600
        ):
            raise ValueError("server.queue_timeout_seconds must be between 0 and 600")


@dataclass(frozen=True)
class AppSettings:
    model_id: str = DEFAULT_MODEL_ID
    runtime: str = "transformers"
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    system_prompt: str = "You are a helpful assistant."
    max_history_turns: int = 20

    def __post_init__(self):
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not self.runtime.strip():
            raise ValueError("runtime must not be empty")
        if not 1 <= self.max_history_turns <= 100:
            raise ValueError("max_history_turns must be between 1 and 100")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    """Load settings, defaulting to the repository's ``user_config.yaml``.

    ``active_model`` and ``generation.max_tokens`` remain accepted for
    compatibility with the existing config file.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        if config_path is not None:
            raise FileNotFoundError(f"Configuration file not found: {path}")
        return AppSettings()

    with path.open("r", encoding="utf-8") as config_file:
        raw = _mapping(yaml.safe_load(config_file), "config")

    generation = _mapping(raw.get("generation"), "generation")
    server = _mapping(raw.get("server"), "server")
    configured_hosts = server.get("allowed_hosts", ("localhost", "127.0.0.1", "::1"))
    if isinstance(configured_hosts, str):
        configured_hosts = configured_hosts.split(",")
    if not isinstance(configured_hosts, (list, tuple)):
        raise ValueError("server.allowed_hosts must be a list of host names")
    model_id = raw.get("model_id", raw.get("active_model", DEFAULT_MODEL_ID))
    max_new_tokens = generation.get(
        "max_new_tokens", generation.get("max_tokens", 2048)
    )

    try:
        generation_settings = GenerationSettings(
            temperature=float(generation.get("temperature", 0.8)),
            top_p=float(generation.get("top_p", 0.95)),
            max_new_tokens=int(max_new_tokens),
        )
        server_settings = ServerSettings(
            host=os.environ.get("LOCAL_LLM_HOST", str(server.get("host", "127.0.0.1"))),
            port=int(os.environ.get("LOCAL_LLM_PORT", server.get("port", 8080))),
            allowed_hosts=tuple(
                host.strip()
                for host in os.environ.get(
                    "LOCAL_LLM_ALLOWED_HOSTS", ",".join(configured_hosts)
                ).split(",")
            ),
            max_request_bytes=int(server.get("max_request_bytes", 4_194_304)),
            max_message_chars=int(server.get("max_message_chars", 262_144)),
            max_queued_requests=int(server.get("max_queued_requests", 4)),
            queue_timeout_seconds=float(server.get("queue_timeout_seconds", 30.0)),
        )
        return AppSettings(
            model_id=str(model_id),
            runtime=str(raw.get("runtime", "transformers")),
            generation=generation_settings,
            server=server_settings,
            system_prompt=str(
                raw.get("system_prompt", "You are a helpful assistant.")
            ),
            max_history_turns=int(raw.get("max_history_turns", 20)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid configuration in {path}: {exc}") from exc
