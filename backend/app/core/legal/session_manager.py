from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SessionRecord:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """
    In-memory session manager.

    Provides a minimal interface for storing session-scoped metadata.
    """

    def __init__(self, ttl_seconds: Optional[int] = None) -> None:
        self._sessions: Dict[str, SessionRecord] = {}
        self._ttl_seconds = ttl_seconds

    def get(self, session_id: str) -> SessionRecord:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionRecord(session_id=session_id)
        rec = self._sessions[session_id]
        rec.last_active = time.time()
        self._evict_expired()
        return rec

    def set(self, session_id: str, key: str, value: Any) -> None:
        rec = self.get(session_id)
        rec.data[key] = value

    def get_value(self, session_id: str, key: str, default: Any = None) -> Any:
        rec = self.get(session_id)
        return rec.data.get(key, default)

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

    def _evict_expired(self) -> None:
        if self._ttl_seconds is None:
            return
        now = time.time()
        expired = [
            sid for sid, rec in self._sessions.items()
            if (now - rec.last_active) > self._ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
