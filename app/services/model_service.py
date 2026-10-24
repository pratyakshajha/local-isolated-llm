"""Model lifecycle and inference service."""

from __future__ import annotations

from threading import Event, Lock, RLock
from typing import Sequence

from app.core.config import AppSettings, GenerationSettings
from app.registry.model_registry import ModelRegistry, ModelSpec
from app.registry.runtime_registry import RuntimeRegistry
from app.runtimes.base import Runtime


class ModelBusyError(RuntimeError):
    """Raised rather than queueing unbounded concurrent model requests."""


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
        self._operation_gate = Lock()

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
        if not self._operation_gate.acquire(blocking=False):
            raise ModelBusyError("The model is currently busy")
        try:
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
        finally:
            self._operation_gate.release()

    def list_models(self) -> list[ModelSpec]:
        return self.model_registry.list_models()

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        generation: GenerationSettings | None = None,
        cancel_event: Event | None = None,
    ) -> str:
        if not self._operation_gate.acquire(blocking=False):
            raise ModelBusyError("The model is already generating a response")
        try:
            with self._lock:
                if not self._runtime.is_loaded:
                    self._runtime.load()
                return self._runtime.generate(
                    messages,
                    generation or self.settings.generation,
                    cancel_event=cancel_event,
                )
        finally:
            self._operation_gate.release()
