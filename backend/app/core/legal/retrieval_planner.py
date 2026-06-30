from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .query_validator import RetrievalStrategy, QueryIntent


@dataclass(frozen=True)
class RetrievalPlan:
    strategy: RetrievalStrategy
    top_k: int
    multi_queries: List[str]
    filters: Dict[str, Any]
    notes: str = ""


class RetrievalPlanner:
    """
    Creates a retrieval plan from query/intents.

    The current backend has retriever.retrieve(query) and expects raw matches,
    so this planner is designed to be non-breaking and optional.
    """

    def __init__(self, default_top_k: int = 5) -> None:
        self.default_top_k = default_top_k

    def plan(
        self,
        query: str,
        intent: Optional[QueryIntent] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.RAG,
        keywords: Optional[List[str]] = None,
    ) -> RetrievalPlan:
        keywords = keywords or []

        # Heuristic defaults.
        top_k = self.default_top_k
        multi_queries: List[str] = []

        if strategy in {RetrievalStrategy.MULTI_DOCUMENT, RetrievalStrategy.HYBRID_SEARCH}:
            top_k = max(self.default_top_k, 8)

        if intent in {QueryIntent.RISK_ANALYSIS, QueryIntent.CASE_ANALYSIS, QueryIntent.DECISION_SUPPORT}:
            top_k = max(top_k, 8)

        # Create very small "multi query" set by appending keywords.
        if keywords:
            # Take up to 3 keywords for query expansion.
            chosen = keywords[:3]
            multi_queries = [f"{query} {kw}" for kw in chosen]

        filters: Dict[str, Any] = {}

        return RetrievalPlan(
            strategy=strategy,
            top_k=top_k,
            multi_queries=multi_queries,
            filters=filters,
            notes="heuristic_plan",
        )
