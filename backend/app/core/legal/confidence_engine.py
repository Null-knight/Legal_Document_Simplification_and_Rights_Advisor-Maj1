from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ConfidenceResult:
    confidence: float
    signals: Dict[str, Any]


class ConfidenceEngine:
    """
    Computes a heuristic confidence score from retrieved evidence.

    This is optional and non-blocking. It provides a stable interface
    for future integration with QueryValidator confidence outputs.
    """

    def compute(
        self,
        query: str,
        retrieved: Optional[List[Dict[str, Any]]],
        intent: Optional[str] = None,
        expected_keywords: Optional[List[str]] = None,
    ) -> ConfidenceResult:
        retrieved = retrieved or []
        expected_keywords = expected_keywords or []

        if not retrieved:
            return ConfidenceResult(confidence=0.0, signals={"reason": "no_retrieved"})

        # Similarity-based signal (assumes retriever may attach "similarity")
        sims: List[float] = []
        for item in retrieved:
            v = item.get("similarity")
            try:
                if v is not None:
                    sims.append(float(v))
            except Exception:
                continue

        avg_sim = sum(sims) / len(sims) if sims else 0.0
        sim_signal = min(1.0, avg_sim / 1.0)  # similarity might already be 0-1 or else; keep safe.

        # Keyword coverage signal
        content_text = " ".join(str(x.get("content", "")) for x in retrieved).lower()
        hit_count = 0
        for kw in expected_keywords:
            kw_l = (kw or "").lower().strip()
            if kw_l and kw_l in content_text:
                hit_count += 1
        denom = max(1, len(expected_keywords))
        keyword_coverage = hit_count / denom

        # Combine
        confidence = 0.55 * sim_signal + 0.45 * keyword_coverage
        confidence = max(0.0, min(1.0, confidence))

        return ConfidenceResult(
            confidence=confidence,
            signals={
                "avg_similarity": avg_sim,
                "sim_signal": sim_signal,
                "keyword_coverage": keyword_coverage,
                "intent": intent,
            },
        )
