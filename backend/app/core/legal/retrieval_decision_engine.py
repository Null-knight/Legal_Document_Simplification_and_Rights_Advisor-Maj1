from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .query_validator import ValidationAction, RetrievalStrategy, QueryIntent
from .query_validator import QueryValidator, ValidationResult


@dataclass(frozen=True)
class RetrievalDecision:
    should_retrieve: bool
    retrieval_strategy: RetrievalStrategy
    top_k: int
    reason: str
    meta: Dict[str, Any]


class RetrievalDecisionEngine:
    """
    Decides whether to retrieve and which retrieval strategy/top_k to use.

    This is designed to be non-breaking and heuristic.
    It uses QueryValidator's existing retrieval strategy logic when available.
    """

    def __init__(self, default_top_k: int = 6) -> None:
        self.default_top_k = default_top_k

    def decide(
        self,
        validation: ValidationResult,
        kb_has_knowledge: Optional[bool] = None,
        requested_top_k: Optional[int] = None,
    ) -> RetrievalDecision:
        action = validation.action

        # Greeting/goodbye/clarify/reject should never retrieve.
        if action in {
            ValidationAction.GREETING,
            ValidationAction.GOODBYE,
            ValidationAction.REJECT,
            ValidationAction.CLARIFY,
        }:
            return RetrievalDecision(
                should_retrieve=False,
                retrieval_strategy=RetrievalStrategy.NO_RETRIEVAL,
                top_k=0,
                reason=f"action={action.value}",
                meta={"detected_domain": validation.detected_domain, "detected_intent": validation.detected_intent.value if validation.detected_intent else None},
            )

        if kb_has_knowledge is None:
            kb_has_knowledge = validation.kb_has_knowledge

        if kb_has_knowledge is False:
            return RetrievalDecision(
                should_retrieve=False,
                retrieval_strategy=RetrievalStrategy.NO_RETRIEVAL,
                top_k=0,
                reason="kb_has_knowledge=false",
                meta={"kb_has_knowledge": kb_has_knowledge},
            )

        # If query validator already picked a strategy, reuse it.
        strategy = getattr(validation, "retrieval_strategy", RetrievalStrategy.NO_RETRIEVAL)
        if strategy == RetrievalStrategy.NO_RETRIEVAL:
            return RetrievalDecision(
                should_retrieve=False,
                retrieval_strategy=RetrievalStrategy.NO_RETRIEVAL,
                top_k=0,
                reason="retrieval_strategy=no_retrieval",
                meta={"retrieval_strategy": strategy.value},
            )

        # Choose top_k with light heuristics.
        top_k = requested_top_k or self.default_top_k
        intent: Optional[QueryIntent] = getattr(validation, "detected_intent", None)
        if intent and intent in {QueryIntent.CASE_ANALYSIS, QueryIntent.DECISION_SUPPORT, QueryIntent.RISK_ANALYSIS}:
            top_k = max(top_k, 8)

        # If the strategy is multi-doc/hybrid, bump top_k.
        if strategy in {RetrievalStrategy.MULTI_DOCUMENT, RetrievalStrategy.HYBRID_SEARCH}:
            top_k = max(top_k, 8)

        return RetrievalDecision(
            should_retrieve=True,
            retrieval_strategy=strategy,
            top_k=top_k,
            reason="validated_and_policy_allows_retrieval",
            meta={
                "retrieval_strategy": strategy.value,
                "detected_domain": validation.detected_domain,
                "detected_intent": intent.value if intent else None,
                "confidence": float(getattr(validation.confidence, "final_confidence", 0.0) if hasattr(validation, "confidence") else 0.0),
            },
        )
