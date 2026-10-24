"""Interface implemented by model inference backends."""

from abc import ABC, abstractmethod
from threading import Event
from typing import Sequence

from app.core.config import GenerationSettings


class Runtime(ABC):
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether this runtime currently has a loaded model."""

    @abstractmethod
    def load(self) -> None:
        """Load model resources."""

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
