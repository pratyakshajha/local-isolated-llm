"""Model lifecycle and inference service."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from threading import Condition, Event, RLock
from time import monotonic
from typing import Iterator, Sequence

from app.core.config import AppSettings, GenerationSettings
from app.registry.model_registry import ModelRegistry, ModelSpec
from app.registry.runtime_registry import RuntimeRegistry
from app.runtimes.base import Runtime


class ModelBusyError(RuntimeError):
    """Raised when the bounded model request queue is full or times out."""


class ModelService:
    def __init__(
        self,
        settings: AppSettings,
        model_registry: ModelRegistry | None = None,
        runtime_registry: RuntimeRegistry | None = None,
    ):
        self.settings = settings
        self.model_registry = model_registry or ModelRegistry()
        self.runtime_registry = runtime_registry or RuntimeRegistry()
        self._model_spec = self.model_registry.resolve(settings.model_id)
        runtime_name = settings.runtime or self._model_spec.runtime
        self._runtime: Runtime = self.runtime_registry.create(
            runtime_name, self._model_spec
        )
        self._lock = RLock()
        self._operation_condition = Condition()
        self._operation_queue: deque[object] = deque()
        self._operation_active = False

    @property
    def model_name(self) -> str:
        return self._model_spec.display_name

    @property
    def model_id(self) -> str:
        return self._model_spec.model_id

    @property
    def is_loaded(self) -> bool:
        return self._runtime.is_loaded

    def load_model(self) -> str:
        with self._lock:
            self._runtime.load()
            return self._model_spec.model_id

    def unload_model(self) -> None:
        with self._lock:
            self._runtime.unload()

    def switch_model(self, model_name: str) -> str:
        model_spec = self.model_registry.resolve(model_name)
        runtime_name = self.settings.runtime or model_spec.runtime
        with self._acquire_operation_slot(timeout=0):
            with self._lock:
                if model_spec.model_id == self._model_spec.model_id:
                    return self._model_spec.model_id

                new_runtime = self.runtime_registry.create(runtime_name, model_spec)
                previous_runtime = self._runtime
                was_loaded = previous_runtime.is_loaded
                if was_loaded:
                    previous_runtime.unload()
                    try:
                        new_runtime.load()
                    except Exception:
                        previous_runtime.load()
                        raise
                self._model_spec = model_spec
                self._runtime = new_runtime
                return self._model_spec.model_id

    @contextmanager
    def _acquire_operation_slot(
        self, timeout: float | None = None, cancel_event: Event | None = None
    ) -> Iterator[None]:
        timeout = (
            self.settings.server.queue_timeout_seconds if timeout is None else timeout
        )
        deadline = monotonic() + timeout
        ticket = object()
        with self._operation_condition:
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Generation was cancelled")
            if not self._operation_active and not self._operation_queue:
                self._operation_active = True
            else:
                if timeout == 0 or (
                    len(self._operation_queue)
                    >= self.settings.server.max_queued_requests
                ):
                    raise ModelBusyError("The model is currently busy")
                self._operation_queue.append(ticket)
                try:
                    while (
                        self._operation_active or self._operation_queue[0] is not ticket
                    ):
                        if cancel_event and cancel_event.is_set():
                            raise InterruptedError("Generation was cancelled")
                        remaining = deadline - monotonic()
                        if remaining <= 0:
                            raise ModelBusyError("The model is currently busy")
                        self._operation_condition.wait(
                            min(remaining, 0.05) if cancel_event else remaining
                        )
                except BaseException:
                    self._operation_queue.remove(ticket)
                    self._operation_condition.notify_all()
                    raise
                self._operation_queue.popleft()
                self._operation_active = True
        try:
            yield
        finally:
            with self._operation_condition:
                self._operation_active = False
                self._operation_condition.notify_all()

    def list_models(self) -> list[ModelSpec]:
        return self.model_registry.list_models()

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        generation: GenerationSettings | None = None,
        cancel_event: Event | None = None,
    ) -> str:
        with self._acquire_operation_slot():
            with self._lock:
                if not self._runtime.is_loaded:
                    self._runtime.load()
                return self._runtime.generate(
                    messages,
                    generation or self.settings.generation,
                    cancel_event=cancel_event,
                )

    def generate_stream(
        self,
        messages: Sequence[dict[str, str]],
        generation: GenerationSettings | None = None,
        cancel_event: Event | None = None,
    ) -> Iterator[str]:
        try:
            with self._acquire_operation_slot(cancel_event=cancel_event):
                with self._lock:
                    if not self._runtime.is_loaded:
                        self._runtime.load()
                    yield from self._runtime.generate_stream(
                        messages,
                        generation or self.settings.generation,
                        cancel_event=cancel_event,
                    )
        except InterruptedError:
            if cancel_event is None or not cancel_event.is_set():
                raise
