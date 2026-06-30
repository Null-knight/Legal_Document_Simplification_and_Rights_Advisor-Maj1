from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .query_validator import QueryIntent


@dataclass(frozen=True)
class KnowledgeVerificationResult:
    verified: bool
    missing: List[str]
    coverage_score: float
    notes: str = ""
    details: Optional[Dict[str, Any]] = None


class KnowledgeVerifier:
    """
    Verifies whether retrieved knowledge is likely sufficient for the intent.

    This is a conservative heuristic-based verifier: it never blocks retrieval,
    but can be used by higher layers to decide formatting, clarification,
    or fallback strategies.
    """

    def __init__(self, min_coverage: float = 0.3) -> None:
        self.min_coverage = min_coverage

    def verify(
        self,
        query: str,
        intent: Optional[QueryIntent],
        retrieved_contents: List[Dict[str, Any]],
        expected_keywords: Optional[List[str]] = None,
    ) -> KnowledgeVerificationResult:
        if expected_keywords is None:
            expected_keywords = []

        query_lower = (query or "").lower().strip()
        contents_text = " ".join(str(item.get("content", "")) for item in retrieved_contents).lower()

        matches = 0
        for kw in expected_keywords:
            kw_l = (kw or "").lower().strip()
            if kw_l and kw_l in contents_text:
                matches += 1

        denom = max(1, len(expected_keywords))
        coverage = matches / denom

        missing = []
        for kw in expected_keywords:
            kw_l = (kw or "").lower().strip()
            if kw_l and kw_l not in contents_text:
                missing.append(kw)

        # Additional weak checks: if no content, fail.
        if not retrieved_contents:
            return KnowledgeVerificationResult(
                verified=False,
                missing=expected_keywords,
                coverage_score=0.0,
                notes="no_retrieved_content",
                details={"query": query_lower, "intent": intent.value if intent else None},
            )

        # Intent keyword heuristics (very light).
        if intent is not None and intent.value and intent.value in contents_text:
            coverage = min(1.0, coverage + 0.15)

        verified = coverage >= self.min_coverage

        return KnowledgeVerificationResult(
            verified=verified,
            missing=missing[:8],
            coverage_score=coverage,
            notes="heuristic_verification",
            details={"matches": matches, "denom": denom, "intent": intent.value if intent else None},
        )
