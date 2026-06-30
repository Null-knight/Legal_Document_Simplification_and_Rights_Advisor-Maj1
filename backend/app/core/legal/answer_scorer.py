from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AnswerScore:
    answered: bool
    covered_all_requested_aspects: bool
    citations_included: bool
    completeness: float
    issues: List[str]
    score: float


class AnswerScorer:
    """
    Scores whether the LLM answer is likely to satisfy the user and whether
    key supporting pieces (citations) are present.

    This is heuristic-only (no LLM calls), designed for immediate integration.
    """

    def score(
        self,
        query: str,
        answer: Optional[str],
        retrieved: Optional[List[Dict[str, Any]]] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        expected_keywords: Optional[List[str]] = None,
    ) -> AnswerScore:
        retrieved = retrieved or []
        citations = citations or []
        expected_keywords = expected_keywords or []

        issues: List[str] = []

        answer_text = (answer or "").strip()
        if not answer_text:
            return AnswerScore(
                answered=False,
                covered_all_requested_aspects=False,
                citations_included=False,
                completeness=0.0,
                issues=["empty_answer"],
                score=0.0,
            )

        # Length / verbosity sanity
        words = answer_text.split()
        if len(words) < 30:
            issues.append("too_short")

        # Hallucination-ish markers
        hallucination_markers = [
            "i am an ai",
            "as an ai",
            "as a language model",
            "my training data",
            "my knowledge cutoff",
            "i cannot browse",
            "i don't have access",
        ]
        a_lower = answer_text.lower()
        if any(m in a_lower for m in hallucination_markers):
            issues.append("hallucination_marker_found")

        # Citation presence
        citations_included = len(citations) > 0 or ("source" in a_lower or "sources" in a_lower or "confidence" in a_lower)
        if not citations_included:
            issues.append("missing_citations_mentions")

        # Keyword coverage from retrieved/context
        content_text = " ".join(str(x.get("content", "")) for x in retrieved).lower()
        expected_hits = 0
        expected = [k.lower().strip() for k in expected_keywords if k and k.strip()]
        for kw in expected:
            if kw in a_lower or kw in content_text:
                expected_hits += 1

        denom = max(1, len(expected))
        keyword_coverage = expected_hits / denom

        # Completeness: combine length + keyword coverage + structure hints
        has_procedural = any(tok in a_lower for tok in ["step", "procedure", "timeline", "documents", "where to file", "how to file"])
        has_rights = any(tok in a_lower for tok in ["your rights", "rights", "entitled", "eligible"])
        structure_bonus = 0.1 if (has_procedural or has_rights) else 0.0

        length_signal = min(1.0, len(words) / 220)  # ~220 words => full
        completeness = max(0.0, min(1.0, (0.45 * length_signal) + (0.45 * keyword_coverage) + structure_bonus))

        # “Answered” decision
        answered = completeness >= 0.55 and "hallucination_marker_found" not in issues
        covered_all_requested_aspects = keyword_coverage >= 0.45

        # Final score
        score = completeness
        if citations and citations_included:
            score = min(1.0, score + 0.10)
        if "too_short" in issues:
            score = max(0.0, score - 0.10)

        return AnswerScore(
            answered=answered,
            covered_all_requested_aspects=covered_all_requested_aspects,
            citations_included=citations_included,
            completeness=completeness,
            issues=issues[:8],
            score=score,
        )
