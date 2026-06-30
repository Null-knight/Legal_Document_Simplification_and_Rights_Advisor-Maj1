from __future__ import annotations

from difflib import SequenceMatcher
import re


class LegalDocumentComparator:
    def compare(self, old_text: str, new_text: str) -> dict[str, list[dict[str, str]]]:
        old_clauses = self._clauses(old_text)
        new_clauses = self._clauses(new_text)
        matcher = SequenceMatcher(a=old_clauses, b=new_clauses, autojunk=False)
        changes = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            changes.append(
                {
                    "change_type": tag,
                    "old": "\n".join(old_clauses[i1:i2]),
                    "new": "\n".join(new_clauses[j1:j2]),
                    "impact_hint": self._impact_hint("\n".join(old_clauses[i1:i2] + new_clauses[j1:j2])),
                }
            )
        return {"changes": changes, "old_clause_count": len(old_clauses), "new_clause_count": len(new_clauses)}

    def _clauses(self, text: str) -> list[str]:
        rough = re.split(r"\n\s*\n|(?<=\.)\s+(?=(?:Clause|Section|\d+\.))", text)
        clauses = [re.sub(r"\s+", " ", clause).strip() for clause in rough if len(clause.strip()) > 15]
        return clauses or [re.sub(r"\s+", " ", text).strip()]

    def _impact_hint(self, text: str) -> str:
        lowered = text.lower()
        if "notice" in lowered or "termination" in lowered:
            return "Review exit rights and notice period."
        if "payment" in lowered or "fee" in lowered or "rent" in lowered:
            return "Review financial obligation change."
        if "liability" in lowered or "indemn" in lowered:
            return "Review risk allocation change."
        if "renew" in lowered:
            return "Review renewal obligation change."
        return "Review legal meaning before signing."
