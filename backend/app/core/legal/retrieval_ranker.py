from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RankedResult:
    items: List[Dict[str, Any]]
    used_signals: Dict[str, Any]


class RetrievalRanker:
    """
    Ranks retrieved documents.

    Current retriever already returns similarity scores (see chat.py usage).
    This ranker is intentionally conservative: it sorts by similarity if present,
    otherwise keeps input order.
    """

    def rank(
        self,
        retrieved: List[Dict[str, Any]],
        query: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RankedResult:
        if not retrieved:
            return RankedResult(items=[], used_signals={"sorted_by": None})

        def _sim(item: Dict[str, Any]) -> float:
            v = item.get("similarity")
            try:
                return float(v) if v is not None else 0.0
            except Exception:
                return 0.0

        sorted_items = sorted(retrieved, key=_sim, reverse=True) if any("similarity" in x for x in retrieved) else list(retrieved)

        if top_k is not None:
            sorted_items = sorted_items[:top_k]

        return RankedResult(
            items=sorted_items,
            used_signals={"sorted_by": "similarity" if any("similarity" in x for x in retrieved) else "none"},
        )
