from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .query_validator import QueryIntent


@dataclass(frozen=True)
class Suggestions:
    items: List[str]


class SuggestionEngine:
    """
    Generates intelligent next questions.

    Heuristic-only (no LLM calls):
    - Uses detected domain + intent to pick a small set of follow-ups
    - If retrieved docs have `metadata.topics`, uses them as additional hints
    """

    def suggest(
        self,
        query: str,
        domain: Optional[str] = None,
        intent: Optional[QueryIntent] = None,
        retrieved_ranked: Optional[List[Dict[str, Any]]] = None,
        max_items: int = 5,
    ) -> Suggestions:
        retrieved_ranked = retrieved_ranked or []

        domain_l = (domain or "").lower().strip()
        intent_v = intent.value if isinstance(intent, QueryIntent) else (str(intent) if intent else "")

        # Base suggestions by common legal domains/intents
        base: List[str] = []
        if domain_l in {"rti", "rti/right to information", "right to information"}:
            base = [
                "How do I file an RTI application?",
                "What are common RTI exemptions and how to challenge them?",
                "RTI timeline: how long does it take to get a response?",
                "RTI appeal process if rejected?",
                "RTI fee structure and how to pay",
            ]
        elif domain_l in {"employment"}:
            base = [
                "What can I do if salary is non-payment?",
                "What is the notice period and termination procedure for employees?",
                "How do I file an employment-related complaint?",
                "What documents should I gather for a salary dispute?",
                "Can I claim damages/compensation for wrongful termination?",
            ]
        elif domain_l in {"tenant", "tenancy"}:
            base = [
                "What are my eviction/lockout rights as a tenant?",
                "How do I respond legally if my landlord tries to evict me illegally?",
                "What notices are required before eviction?",
                "How can I claim refund of security deposit (if applicable)?",
                "How do I file a complaint for tenant rights violations?",
            ]
        elif domain_l in {"contract"}:
            base = [
                "What counts as breach of contract?",
                "What remedies are available for breach (damages/injunction/etc.)?",
                "How do I draft a notice for contract breach?",
                "What evidence should I keep for a contract dispute?",
                "What are essential elements of a valid contract?",
            ]
        elif domain_l in {"consumer"}:
            base = [
                "How do I file a consumer complaint?",
                "What documents are needed for refund/warranty claims?",
                "What are my rights for defective products?",
                "What is the timeline for consumer redressal?",
                "Can I claim compensation for deficiency in service?",
            ]
        else:
            # Generic, but legal-specific
            base = [
                "What legal steps should I take next for this situation?",
                "What documents/evidence should I collect?",
                "Is there a form/notice I should use?",
                "What is the expected timeline?",
                "What are possible outcomes and remedies?",
            ]

        # Intent-driven pruning/boost
        if intent_v == QueryIntent.PROCEDURE.value:
            base = [x for x in base if "how do I file" in x.lower() or "steps" in x.lower() or "timeline" in x.lower()]
        elif intent_v == QueryIntent.RIGHTS.value:
            base = [x for x in base if "rights" in x.lower() or "my rights" in x.lower()]

        # Topics from retrieved metadata as additional options
        topic_suggestions: List[str] = []
        for item in retrieved_ranked[:6]:
            meta = item.get("metadata") or {}
            topics = meta.get("topics") or meta.get("topic") or ""
            if isinstance(topics, str) and topics.strip():
                parts = [t.strip() for t in topics.split(",") if t.strip()]
                for p in parts[:3]:
                    topic_suggestions.append(f"Can you explain '{p}' for my case?")

        # De-dup while preserving order
        combined: List[str] = []
        for s in base + topic_suggestions:
            if s and s not in combined:
                combined.append(s)

        return Suggestions(items=combined[:max_items])
