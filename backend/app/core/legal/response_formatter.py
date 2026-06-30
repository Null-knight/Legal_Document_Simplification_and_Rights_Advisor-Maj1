from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FormattedResponse:
    answer: str
    citations: List[Dict[str, Any]]


class ResponseFormatter:
    """
    Formats an LLM answer + citations into the shape expected by the API.

    The current API (backend/app/api/routes/chat.py) formats citations separately,
    but this module gives a stable place to centralize response formatting.
    """

    def format(
        self,
        answer: str,
        retrieved: Optional[List[Dict[str, Any]]] = None,
        max_citations: int = 5,
        truncate_chars: int = 350,
    ) -> FormattedResponse:
        answer = answer or ""

        citations: List[Dict[str, Any]] = []
        retrieved = retrieved or []

        for i, item in enumerate(retrieved[:max_citations]):
            metadata = item.get("metadata") or {}
            content = str(item.get("content") or "")
            title = metadata.get("title") or metadata.get("filename") or metadata.get("source") or "Local legal source"
            citations.append(
                {
                    "source": f"Source {i + 1}",
                    "content": content[:truncate_chars],
                    "metadata": metadata,
                    "similarity": item.get("similarity"),
                    "title": title,
                }
            )

        return FormattedResponse(answer=answer, citations=citations)
