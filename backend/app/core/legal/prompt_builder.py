from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .constants import DEFAULT_SYSTEM_INSTRUCTION


@dataclass(frozen=True)
class PromptBuildResult:
    system: str
    user: str
    metadata: Dict[str, Any]


class PromptBuilder:
    """
    Builds LLM prompts for legal QA/rag-style answers.

    This project already has prompt templates in app/core/llm/prompts.py.
    This module provides a consistent adapter for future pipeline usage.
    """

    def build_rag_prompt(
        self,
        query: str,
        context: str,
        disclaimer: Optional[str] = None,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        extra: Optional[Dict[str, Any]] = None,
    ) -> PromptBuildResult:
        disclaimer = disclaimer or ""
        extra = extra or {}

        system = system_instruction.strip()

        user_parts = [
            "Use the context below as legal evidence. Do not copy it directly.",
            "Write a natural answer that fits the user's intent.",
            "Avoid repetitive headings such as 'Confidence' unless the user asks for technical scoring.",
            "",
            f"Question:\n{query.strip()}",
            "",
            "Retrieved legal context:",
            context.strip() if context else "(no context)",
        ]

        if disclaimer:
            user_parts.extend(["", "Disclaimer:", disclaimer.strip()])

        user_parts.append("")
        user = "\n".join(user_parts)

        return PromptBuildResult(
            system=system,
            user=user,
            metadata={"query": query, **extra},
        )
