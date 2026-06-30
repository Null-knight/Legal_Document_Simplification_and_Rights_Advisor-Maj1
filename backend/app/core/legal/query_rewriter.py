from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, List

from .query_validator import ExtractedEntities, ConversationTracker, QueryIntent


@dataclass(frozen=True)
class RewrittenQuery:
    query: str
    keywords: List[str]
    rationale: str = ""


class QueryRewriter:
    """
    Rewrites incoming user queries to improve retrieval.

    This module is intentionally lightweight and dependency-free so it can
    be introduced without breaking existing APIs.
    """

    def __init__(self) -> None:
        # Common lightweight cleanup patterns (do not over-aggressively rewrite)
        self._whitespace_re = re.compile(r"\s+")
        self._punct_re = re.compile(r"[“”\"']")

    def rewrite(
        self,
        query: str,
        session: Optional[ConversationTracker] = None,
        entities: Optional[ExtractedEntities] = None,
        detected_intent: Optional[QueryIntent] = None,
    ) -> RewrittenQuery:
        if not query:
            return RewrittenQuery(query="", keywords=[], rationale="empty_input")

        cleaned = query.strip()
        cleaned = self._punct_re.sub("", cleaned)
        cleaned = self._whitespace_re.sub(" ", cleaned).strip()

        # If we have an existing conversation tracker, leverage its follow-up resolution.
        if session is not None and hasattr(session, "is_followup") and session.is_followup(cleaned):
            try:
                cleaned = session.resolve_followup(cleaned)
            except Exception:
                # Never fail the pipeline due to rewriting.
                pass

        keywords: List[str] = []
        if entities is not None:
            keywords = self._extract_keywords_from_entities(entities)

        # Add a few query tokens as retrieval keywords.
        keywords.extend(self._extract_query_tokens(cleaned))
        keywords = list(dict.fromkeys([k for k in keywords if k]))[:12]

        rationale_parts = ["cleaned_input"]
        if session is not None:
            rationale_parts.append("followup_context_if_applicable")
        if detected_intent is not None:
            rationale_parts.append(f"intent={detected_intent.value}")

        return RewrittenQuery(query=cleaned, keywords=keywords, rationale=";".join(rationale_parts))

    def _extract_keywords_from_entities(self, entities: ExtractedEntities) -> List[str]:
        out: List[str] = []
        out.extend(entities.acts or [])
        out.extend(entities.sections or [])
        out.extend(entities.courts or [])
        out.extend(entities.authorities or [])
        out.extend(entities.legal_concepts or [])
        out.extend(entities.rights or [])
        return out

    def _extract_query_tokens(self, query: str) -> List[str]:
        tokens = re.findall(r"\b[\w-]+\b", query.lower())
        stopwords = {
            "what", "is", "are", "am", "the", "a", "an", "to", "for", "of", "with", "on", "in",
            "and", "or", "how", "why", "when", "where", "who", "please", "tell", "explain",
            "can", "could", "should", "would"
        }
        keywords = [t for t in tokens if t not in stopwords and len(t) > 2]
        return keywords
