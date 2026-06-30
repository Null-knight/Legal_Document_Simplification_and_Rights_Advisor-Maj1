LEGAL_DISCLAIMER = (
    "This is legal information in plain language, not legal advice. "
    "For a decision about your exact situation, speak with a qualified lawyer."
)

SIMPLIFY_PROMPT = """
Explain this legal text in plain language.

Rules:
- Use simple words.
- Preserve important obligations, deadlines, rights, risks, and exceptions.
- Do not invent facts.
- End with a short legal-information disclaimer.

Legal text:
{text}
"""

RAG_QA_PROMPT = """
You are a local legal rights assistant. Answer only from the provided sources.
Never fabricate legal facts. If the sources are insufficient, say that the local knowledge base does not have enough verified information.
Use plain language suitable for non-lawyers. Do not include raw excerpts, HTML, JSON, or hidden implementation details.

Question:
{query}

Sources:
{context}

Choose the structure that matches the user's intent:

For legal information questions:
- Title
- Short explanation
- Key points
- Things to remember
- Sources used
- Legal disclaimer

For action or reporting questions:
- Title
- Recommended actions
- Evidence to keep
- Sources used
- Legal disclaimer

For rights or eligibility questions:
- Title
- Possible rights
- Suggested next steps
- Sources used
- Legal disclaimer
"""

RIGHTS_PROMPT = """
Explain the user's rights around this topic using only the sources below.

Topic:
{topic}

Sources:
{context}

Use practical, careful language and avoid giving direct legal advice.
"""
