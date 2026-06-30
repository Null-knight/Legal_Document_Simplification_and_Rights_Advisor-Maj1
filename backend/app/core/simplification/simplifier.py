from __future__ import annotations

import re

from app.core.llm.llm_manager import LLMManager
from app.core.llm.prompts import LEGAL_DISCLAIMER, SIMPLIFY_PROMPT


class LegalSimplifier:
    def __init__(self, llm_manager: LLMManager | None = None) -> None:
        self.llm_manager = llm_manager or LLMManager()

    async def simplify(self, text: str) -> str:
        llm_answer = await self.llm_manager.generate(SIMPLIFY_PROMPT.format(text=text[:12000]))
        if llm_answer:
            return llm_answer
        return self._rule_based_simplification(text)

    def _rule_based_simplification(self, text: str) -> str:
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
        important = [
            sentence
            for sentence in sentences
            if re.search(r"\b(shall|must|required|may|terminate|notice|liable|penalty|right|consent)\b", sentence, re.I)
        ]
        sample = important[:8] or sentences[:8]
        bullets = "\n".join(f"- {self._plain_rewrite(sentence)}" for sentence in sample)
        return f"Plain-language highlights:\n{bullets}\n\n{LEGAL_DISCLAIMER}"

    def _plain_rewrite(self, sentence: str) -> str:
        replacements = {
            "shall": "must",
            "hereby": "",
            "thereof": "of it",
            "pursuant to": "under",
            "prior to": "before",
            "commence": "start",
            "terminate": "end",
        }
        rewritten = sentence
        for old, new in replacements.items():
            rewritten = re.sub(rf"\b{old}\b", new, rewritten, flags=re.I)
        return re.sub(r"\s+", " ", rewritten).strip()
