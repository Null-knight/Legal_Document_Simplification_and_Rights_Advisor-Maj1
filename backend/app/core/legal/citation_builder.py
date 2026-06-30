from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CitationDraft:
    source: str
    content: str
    metadata: Dict[str, Any]
    similarity: Optional[float] = None


class CitationBuilder:
    """
    Builds citations from retrieved documents.

    This is extracted from backend/app/api/routes/chat.py so citation styles
    can evolve independently from retrieval and formatting.
    """

    def build(
        self,
        retrieved_ranked: List[Dict[str, Any]],
        max_citations: int = 5,
        truncate_chars: int = 350,
    ) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []

        for index, item in enumerate((retrieved_ranked or [])[:max_citations]):
            metadata = item.get("metadata") or {}
            title = (
                metadata.get("title")
                or metadata.get("filename")
                or metadata.get("source")
                or f"Source {index + 1}"
            )

            citations.append(
                {
                    "source": title,
                    "content": str(item.get("content") or "")[:truncate_chars],
                    "metadata": metadata,
                    "similarity": item.get("reranked_score", item.get("similarity")),
                    "title": title,
                }
            )

        return citations
