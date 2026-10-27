"""Interface implemented by model inference backends."""

from abc import ABC, abstractmethod
from threading import Event
from typing import Callable, Iterator, Sequence

from app.core.config import GenerationSettings


class Runtime(ABC):
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether this runtime currently has a loaded model."""

    @abstractmethod
    def load(self) -> None:
        """Load model resources."""

    def prefetch(
        self, progress: Callable[[int, int | None], None] | None = None
    ) -> None:
        """Download model resources before loading them, if the runtime supports it."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""

    @abstractmethod
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        settings: GenerationSettings,
        cancel_event: Event | None = None,
    ) -> str:
        """Generate one assistant response, stopping if cancellation is requested."""

    def generate_stream(
        self,
        messages: Sequence[dict[str, str]],
        settings: GenerationSettings,
        cancel_event: Event | None = None,
    ) -> Iterator[str]:
        """Stream response text, falling back to a single generated chunk."""
        response = self.generate(messages, settings, cancel_event=cancel_event)
        if response:
            yield response

    def generate_completion(
        self,
        prompt: str,
        settings: GenerationSettings,
        cancel_event: Event | None = None,
    ) -> str:
        """Generate from a plain prompt when a runtime has no native path."""
        return self.generate(
            [{"role": "user", "content": prompt}],
            settings,
            cancel_event=cancel_event,
        )

    def generate_completion_stream(
        self,
        prompt: str,
        settings: GenerationSettings,
        cancel_event: Event | None = None,
    ) -> Iterator[str]:
        """Stream a plain-prompt response when a runtime has no native path."""
        yield from self.generate_stream(
            [{"role": "user", "content": prompt}],
            settings,
            cancel_event=cancel_event,
        )
