from __future__ import annotations

import re

from app.core.rag.embeddings import EmbeddingManager


CLAUSE_PATTERNS = {
    "parties": r"\b(between|party|parties|lessor|lessee|employer|employee|client|vendor)\b",
    "duration": r"\b(term|duration|effective date|expiry|expires|period of)\b",
    "payment_terms": r"\b(payment|fee|salary|rent|invoice|consideration|deposit)\b",
    "termination": r"\b(terminate|termination|notice period|end this agreement|breach)\b",
    "penalty": r"\b(penalty|late fee|liquidated damages|fine|forfeit)\b",
    "liability": r"\b(liability|liable|responsible|damages|losses)\b",
    "indemnity": r"\b(indemnify|indemnity|hold harmless)\b",
    "arbitration": r"\b(arbitration|arbitrator|conciliation|dispute resolution)\b",
    "auto_renewal": r"\b(auto(?:matic)? renewal|renew automatically|unless terminated|successive term)\b",
    "confidentiality": r"\b(confidential|non-disclosure|proprietary information)\b",
    "governing_law": r"\b(governing law|jurisdiction|courts at|venue)\b",
}

CLAUSE_PROTOTYPES = {
    "termination": [
        "Either party may end this agreement after giving written notice.",
        "This contract shall cease after a specified period.",
        "The arrangement may be discontinued for breach or non-performance.",
    ],
    "penalty": [
        "A party must pay a penalty or late fee for delay or breach.",
        "Liquidated damages are payable on default.",
    ],
    "liability": [
        "A party is responsible for losses, claims, damages, or legal costs.",
        "Liability may be limited, capped, excluded, or unlimited.",
    ],
    "indemnity": [
        "One party will indemnify and hold the other harmless from claims.",
        "A party must compensate the other for third-party losses.",
    ],
    "arbitration": [
        "Disputes will be resolved by arbitration or an arbitrator.",
        "The parties agree to alternative dispute resolution.",
    ],
    "auto_renewal": [
        "The agreement renews automatically unless notice is given.",
        "The term continues for successive renewal periods.",
    ],
    "payment_terms": [
        "Payment, rent, fees, invoices, salary, or deposits are due.",
        "A party must pay an amount by a specific date.",
    ],
}

SEMANTIC_THRESHOLDS = {
    "termination": 0.48,
    "payment_terms": 0.52,
    "liability": 0.54,
    "penalty": 0.56,
    "indemnity": 0.56,
    "arbitration": 0.58,
    "auto_renewal": 0.72,
}


class ClauseExtractor:
    def __init__(self) -> None:
        self.embeddings = EmbeddingManager()
        self._prototype_embeddings: dict[str, list[list[float]]] | None = None

    def extract(self, text: str) -> dict[str, list[dict[str, str]]]:
        sentences = self._sentences(text)
        output: dict[str, list[dict[str, str]]] = {name: [] for name in CLAUSE_PATTERNS}
        for sentence in sentences:
            matched = set()
            for clause_type, pattern in CLAUSE_PATTERNS.items():
                if re.search(pattern, sentence, re.I):
                    matched.add(clause_type)
                    output[clause_type].append(
                        {
                            "clause": sentence,
                            "why_detected": pattern,
                            "method": "regex",
                        }
                    )
            for clause_type, score in self._semantic_matches(sentence).items():
                if clause_type in matched:
                    continue
                output.setdefault(clause_type, []).append(
                    {
                        "clause": sentence,
                        "why_detected": f"semantic prototype similarity {score:.2f}",
                        "method": "embedding",
                    }
                )
        return {name: clauses for name, clauses in output.items() if clauses}

    def _sentences(self, text: str) -> list[str]:
        chunks = re.split(r"(?<=[.!?;])\s+|\n+", text)
        return [re.sub(r"\s+", " ", chunk).strip() for chunk in chunks if len(chunk.strip()) > 20]

    def _semantic_matches(self, sentence: str) -> dict[str, float]:
        try:
            self._ensure_prototypes()
            sentence_embedding = self.embeddings.embed_query(sentence)
        except Exception:
            return {}
        matches = {}
        for clause_type, embeddings in (self._prototype_embeddings or {}).items():
            best = max(self._cosine(sentence_embedding, prototype) for prototype in embeddings)
            if best >= SEMANTIC_THRESHOLDS.get(clause_type, 0.55):
                matches[clause_type] = best
        return matches

    def _ensure_prototypes(self) -> None:
        if self._prototype_embeddings is not None:
            return
        self._prototype_embeddings = {
            clause_type: self.embeddings.embed_documents(examples)
            for clause_type, examples in CLAUSE_PROTOTYPES.items()
        }

    def _cosine(self, left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))
