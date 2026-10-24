"""In-memory chat session service, independent of a concrete runtime."""

from dataclasses import dataclass
from threading import Event, RLock
from time import monotonic
from uuid import uuid4

from app.llm.conversation import Conversation
from app.services.model_service import ModelBusyError, ModelService


class SessionLimitError(RuntimeError):
    """Raised when all permitted chat sessions are active."""


@dataclass
class _ChatSession:
    conversation: Conversation
    last_accessed: float
    active_generations: int = 0
    active_connections: int = 0


class ChatService:
    def __init__(
        self,
        model_service: ModelService,
        system_prompt: str = "You are a helpful assistant.",
        max_history_turns: int = 20,
        max_sessions: int = 128,
        session_ttl_seconds: int = 3_600,
    ):
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if session_ttl_seconds < 1:
            raise ValueError("session_ttl_seconds must be at least 1")
        self.model_service = model_service
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.max_sessions = max_sessions
        self.session_ttl_seconds = session_ttl_seconds
        self._sessions: dict[str, _ChatSession] = {}
        self._sessions_lock = RLock()

    @property
    def model_name(self) -> str:
        return self.model_service.model_name

    @property
    def active_sessions(self) -> int:
        now = monotonic()
        with self._sessions_lock:
            self._remove_expired_sessions(now)
            return len(self._sessions)

    def start_session(self, *, retain: bool = False) -> str:
        session_id = str(uuid4())
        now = monotonic()
        with self._sessions_lock:
            self._remove_expired_sessions(now)
            if len(self._sessions) >= self.max_sessions:
                idle_sessions = [
                    (key, session)
                    for key, session in self._sessions.items()
                    if session.active_generations == 0 and session.active_connections == 0
                ]
                if not idle_sessions:
                    raise SessionLimitError("All chat sessions are currently active")
                oldest_id, _ = min(idle_sessions, key=lambda item: item[1].last_accessed)
                self._sessions.pop(oldest_id, None)
            self._sessions[session_id] = _ChatSession(
                Conversation(self.system_prompt, self.max_history_turns),
                now,
                active_connections=int(retain),
            )
        return session_id

    def _remove_expired_sessions(self, now: float) -> None:
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.active_generations == 0
            and session.active_connections == 0
            and now - session.last_accessed >= self.session_ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def close_session(self, session_id: str) -> None:
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

    def generate(
        self,
        session_id: str,
        user_message: str,
        cancel_event: Event | None = None,
    ) -> str | None:
        if not user_message.strip():
            raise ValueError("Message must not be empty")
        now = monotonic()
        with self._sessions_lock:
            self._remove_expired_sessions(now)
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Unknown chat session: {session_id}")
            session.active_generations += 1
            session.last_accessed = now

        try:
            if not session.conversation.lock.acquire(blocking=False):
                raise ModelBusyError("This chat is already generating a response")
            try:
                if cancel_event and cancel_event.is_set():
                    return None
                messages = session.conversation.messages_for(user_message)
                response = self.model_service.generate(messages, cancel_event=cancel_event)
                if cancel_event and cancel_event.is_set():
                    return None
                session.conversation.add_turn(user_message, response)
                return response
            finally:
                session.conversation.lock.release()
        finally:
            with self._sessions_lock:
                session.active_generations -= 1
                session.last_accessed = monotonic()
