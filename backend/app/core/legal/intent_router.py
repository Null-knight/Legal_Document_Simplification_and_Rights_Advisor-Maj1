from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

from .query_validator import QueryIntent


@dataclass(frozen=True)
class RoutedIntent:
    intent: QueryIntent
    route: str
    confidence: float = 0.0
    details: Optional[Dict[str, Any]] = None


class IntentRouter:
    """
    Small adapter that maps detected QueryIntent to an internal "route" name.

    This project currently uses query_validator.py heavily; this router provides
    a stable interface for future pipeline modules without changing existing APIs.
    """

    def route(self, intent: QueryIntent, confidence: float = 0.0, **details: Any) -> RoutedIntent:
        intent_value = intent.value if isinstance(intent, QueryIntent) else str(intent)

        mapping = {
            QueryIntent.RIGHTS.value: "rights",
            QueryIntent.PROCEDURE.value: "procedure",
            QueryIntent.COMPLAINT.value: "complaint",
            QueryIntent.APPEAL.value: "appeal",
            QueryIntent.DOCUMENT.value: "document",
            QueryIntent.DRAFTING.value: "drafting",
            QueryIntent.RISK_ANALYSIS.value: "risk_analysis",
            QueryIntent.COMPARISON.value: "comparison",
            QueryIntent.SUMMARIZATION.value: "summarization",
            QueryIntent.DEFINITION.value: "definition",
            QueryIntent.EXPLANATION.value: "explanation",
        }

        route = mapping.get(intent_value, "general")
        return RoutedIntent(intent=intent, route=route, confidence=confidence, details=details or None)
