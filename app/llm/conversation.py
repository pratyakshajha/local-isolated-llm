"""Bounded, in-memory conversation history."""

from threading import RLock
from typing import Any


class Conversation:
    def __init__(self, system_prompt: str, max_history_turns: int = 20):
        if max_history_turns < 1:
            raise ValueError("max_history_turns must be at least 1")
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self._turns: list[tuple[str, str]] = []
        self.lock = RLock()

    def messages_for(self, user_message: str) -> list[dict[str, str]]:
        with self.lock:
            messages = [{"role": "system", "content": self.system_prompt}]
            for user_text, assistant_text in self._turns:
                messages.extend(
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ]
                )
            messages.append({"role": "user", "content": user_message})
            return messages

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        with self.lock:
            self._turns.append((user_message, assistant_message))
            excess = len(self._turns) - self.max_history_turns
            if excess > 0:
                del self._turns[:excess]

    def history(self) -> list[dict[str, Any]]:
        """Return a copy of completed turns, excluding the system prompt."""
        with self.lock:
            messages: list[dict[str, Any]] = []
            for user_text, assistant_text in self._turns:
                messages.extend(
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ]
                )
            return messages
