"""Runtime factory registry."""

from collections.abc import Callable

from app.registry.model_registry import ModelSpec
from app.runtimes.base import Runtime
from app.runtimes.transformers.runtime import TransformersRuntime


class RuntimeRegistry:
    def __init__(self):
        self._factories: dict[str, Callable[[ModelSpec], Runtime]] = {
            "transformers": lambda model: TransformersRuntime(model.model_id)
        }

    def register(self, name: str, factory: Callable[[ModelSpec], Runtime]) -> None:
        normalized_name = name.strip().casefold()
        if not normalized_name:
            raise ValueError("Runtime name must not be empty")
        self._factories[normalized_name] = factory

    def create(self, name: str, model: ModelSpec) -> Runtime:
        normalized_name = name.strip().casefold()
        try:
            factory = self._factories[normalized_name]
        except KeyError as exc:
            supported = ", ".join(sorted(self._factories))
            raise ValueError(
                f"Unknown runtime {name!r}. Registered runtimes: {supported}"
            ) from exc
        return factory(model)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
