from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BuiltContext:
    context_text: str
    sources: List[Dict[str, Any]]
    context_metadata: Dict[str, Any]


class ContextBuilder:
    """
    Builds a single context string from retrieved documents.

    Expected input: list[dict] items that may contain:
    - content (str)
    - metadata (dict)
    - similarity (float)
    """

    def build(
        self,
        query: str,
        retrieved: List[Dict[str, Any]],
        max_sources: int = 6,
        include_metadata: bool = True,
    ) -> BuiltContext:
        if not retrieved:
            return BuiltContext(
                context_text="",
                sources=[],
                context_metadata={"query": query, "source_count": 0},
            )

        sources: List[Dict[str, Any]] = retrieved[:max_sources]

        parts: List[str] = []
        for i, item in enumerate(sources):
            content = str(item.get("content", "")).strip()
            if not content:
                continue

            meta = item.get("metadata") or {}
            title = meta.get("title") or meta.get("filename") or meta.get("source") or "Local legal source"
            sim = item.get("similarity", None)

            if include_metadata:
                header = f"[Source {i + 1}] {title}"
                if sim is not None:
                    header += f" (similarity={sim})"
                parts.append(header + "\n" + content)
            else:
                parts.append(content)

        context_text = "\n\n".join(parts).strip()

        return BuiltContext(
            context_text=context_text,
            sources=sources,
            context_metadata={
                "query": query,
                "source_count": len(sources),
                "max_sources": max_sources,
                "include_metadata": include_metadata,
            },
        )
