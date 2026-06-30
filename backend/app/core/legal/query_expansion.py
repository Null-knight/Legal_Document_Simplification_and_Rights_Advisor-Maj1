from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ExpandedQuery:
    expanded_query: str
    expansions: List[str]
    keywords: List[str]


class QueryExpansionEngine:
    """
    Expands user queries with legal synonyms/paraphrases before retrieval.

    Lightweight and deterministic (no LLM calls).
    """

    def __init__(self) -> None:
        # Phrase-level expansions (more important than single-token replacements)
        self.phrase_map: Dict[str, List[str]] = {
            # Employment / wages
            "salary stopped": ["salary non payment", "wages due", "unpaid salary", "dues"],
            "salary non payment": ["salary non payment", "wages due", "unpaid salary", "dues"],
            "salary not paid": ["unpaid salary", "wages due", "salary arrears", "dues"],
            "salary unpaid": ["unpaid salary", "wages due", "salary arrears", "dues"],

            # Tenant eviction / illegal eviction
            "owner kicked": ["illegal eviction", "tenant eviction rights", "lockout", "forcible eviction"],
            "owner kicked me out": ["illegal eviction", "tenant eviction rights", "forcible eviction"],
            "kicked me out": ["illegal eviction", "tenant eviction rights", "forcible eviction"],
            "kicked out": ["illegal eviction", "tenant eviction rights", "forcible eviction"],
            "illegal eviction": ["illegal eviction", "tenant eviction rights", "forcible eviction"],
        }

        # Token-level synonym expansions
        self.token_map: Dict[str, List[str]] = {
            "salary": ["unpaid salary", "salary non payment", "wages", "dues"],
            "wages": ["salary", "unpaid salary", "dues"],
            "payment": ["non payment", "non-payment", "arrears", "dues"],
            "eviction": ["illegal eviction", "tenant eviction rights", "forcible eviction"],
            "tenant": ["landlord tenant dispute", "tenant eviction rights"],
            "owner": ["landlord"],
            "kicked": ["eviction"],
            "lockout": ["forcible eviction", "illegal eviction"],
        }

        self.stopwords = {
            "what",
            "is",
            "are",
            "a",
            "an",
            "the",
            "to",
            "for",
            "of",
            "with",
            "on",
            "in",
            "and",
            "or",
            "please",
            "tell",
            "explain",
            "can",
            "could",
            "should",
            "would",
            "me",
            "my",
            "i",
            "you",
            "he",
            "she",
            "they",
            "them",
            "it",
            "this",
            "that",
        }

    def expand(self, query: str, domain: Optional[str] = None) -> ExpandedQuery:
        q = (query or "").strip()
        if not q:
            return ExpandedQuery(expanded_query="", expansions=[], keywords=[])

        q_lower = q.lower()

        expansions: List[str] = []
        # Phrase-level
        for phrase, reps in self.phrase_map.items():
            if phrase in q_lower:
                expansions.extend(reps)

        # Token-level
        tokens = re.findall(r"\b[\w-]+\b", q_lower)
        for t in tokens:
            if t in self.stopwords:
                continue
            if t in self.token_map:
                expansions.extend(self.token_map[t])

        # Domain hinting (optional)
        if domain:
            d = domain.lower()
            if d == "employment" and ("salary" in q_lower or "wages" in q_lower or "payment" in q_lower):
                expansions.extend(["salary non payment", "unpaid salary", "wages due", "arrears"])
            if d in {"tenant", "tenancy"} and ("eviction" in q_lower or "kicked" in q_lower or "owner" in q_lower):
                expansions.extend(["illegal eviction", "tenant eviction rights", "forcible eviction"])

        # De-dup + keywords
        expansions = list(dict.fromkeys([e for e in expansions if e]))
        keywords = self._keywords_from_expansions(expansions)

        # Expanded query string: append expansions if not already present
        expanded_query = q
        if keywords:
            expanded_query = f"{q} " + " ".join(keywords[:12])

        return ExpandedQuery(expanded_query=expanded_query.strip(), expansions=expansions, keywords=keywords)

    def _keywords_from_expansions(self, expansions: List[str]) -> List[str]:
        tokens: List[str] = []
        for e in expansions:
            for t in re.findall(r"\b[\w-]+\b", e.lower()):
                if t in self.stopwords:
                    continue
                if len(t) <= 2:
                    continue
                tokens.append(t)

        # De-dup while preserving order
        out = list(dict.fromkeys(tokens))

        # Prefer returning the full multiword phrases as keywords too
        # (we’ll include phrase tokens already; this is mostly for simple routing)
        return out
