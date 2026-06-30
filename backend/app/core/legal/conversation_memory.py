from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    role: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMemoryState:
    session_id: str
    items: List[MemoryItem] = field(default_factory=list)
    last_updated: float = 0.0


class ConversationMemory:
    """
    Simple in-memory conversation memory store.

    This module does not integrate with persistence; it provides an interface
    compatible with future session persistence.
    """

    def __init__(self) -> None:
        self._store: Dict[str, ConversationMemoryState] = {}

    def get_state(self, session_id: str) -> ConversationMemoryState:
        if session_id not in self._store:
            self._store[session_id] = ConversationMemoryState(session_id=session_id)
        return self._store[session_id]

    def append(self, session_id: str, role: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        state = self.get_state(session_id)
        state.items.append(MemoryItem(role=role, message=message, metadata=metadata or {}))

    def get_history(self, session_id: str, limit: int = 20) -> List[MemoryItem]:
        state = self.get_state(session_id)
        if limit <= 0:
            return []
        return state.items[-limit:]

    def clear(self, session_id: str) -> None:
        if session_id in self._store:
            del self._store[session_id]
