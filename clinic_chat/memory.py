from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from clinic_chat.config import MEMORY_MAX_MESSAGES

Role = Literal["user", "assistant", "system"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class SessionMemory:
    """Memória de curto prazo por sessão (apenas mensagens user/assistant para o LLM)."""

    messages: list[ChatMessage] = field(default_factory=list)

    def append(self, role: Role, content: str) -> None:
        if role == "system":
            return
        self.messages.append(ChatMessage(role=role, content=content))
        self._trim()

    def _trim(self) -> None:
        if len(self.messages) <= MEMORY_MAX_MESSAGES:
            return
        # Mantém as mensagens mais recentes
        self.messages = self.messages[-MEMORY_MAX_MESSAGES:]

    def as_llm_history(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self) -> None:
        self.messages.clear()


class MemoryStore:
    """Armazena memória indexada por (user_id, session_id)."""

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], SessionMemory] = defaultdict(SessionMemory)

    def get(self, user_id: str, session_id: str) -> SessionMemory:
        return self._sessions[(user_id, session_id)]

    def new_session(self, user_id: str, session_id: str) -> None:
        self._sessions[(user_id, session_id)] = SessionMemory()
