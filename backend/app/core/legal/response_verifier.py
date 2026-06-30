from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass(frozen=True)
class ResponseVerificationResult:
    ok: bool
    issues: List[str]
    confidence: float
    normalized_answer: str = ""


class ResponseVerifier:
    """
    Lightweight verifier for assistant responses.

    Designed to be conservative and non-blocking for now.
    """

    def __init__(self, min_confidence: float = 0.35) -> None:
        self.min_confidence = min_confidence

    def verify(self, answer: Optional[str], citations: Optional[List[Dict[str, Any]]] = None) -> ResponseVerificationResult:
        issues: List[str] = []
        answer = (answer or "").strip()
        citations = citations or []

        if not answer:
            return ResponseVerificationResult(ok=False, issues=["empty_answer"], confidence=0.0, normalized_answer="")

        # Basic sanity checks
        if len(answer) < 30:
            issues.append("too_short")
        if "confidence" not in answer.lower():
            issues.append("missing_confidence_section")
        if "source" not in answer.lower() and "citation" not in answer.lower():
            if citations:
                issues.append("missing_source_mentions")

        # If citations exist, raise confidence a bit.
        confidence = 0.4
        if len(citations) > 0:
            confidence += 0.2
        if "confidence" in answer.lower():
            confidence += 0.2
        if len(issues) == 0:
            confidence += 0.2

        confidence = min(1.0, confidence)

        ok = confidence >= self.min_confidence and len(issues) <= 2
        return ResponseVerificationResult(
            ok=ok,
            issues=issues,
            confidence=confidence,
            normalized_answer=answer,
        )
