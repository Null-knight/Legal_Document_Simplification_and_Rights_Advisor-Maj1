from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.core.legal.query_validator import ValidationAction, ValidationResult


@dataclass
class RetrievalPipelineResult:
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    retrieved_matches: List[Dict[str, Any]]


class RetrievalPipeline:
    """Conversation-aware legal RAG controller.

    The pipeline deliberately separates validation, intent routing, domain-aware
    retrieval, relevance checks, deterministic formatting, and citations. This
    keeps general legal explanations from accidentally using the rights template.
    """

    DOMAIN_ALIASES = {
        "tenant": "TENANT",
        "tenancy": "TENANT",
        "rental": "TENANT",
        "consumer": "CONSUMER",
        "employment": "EMPLOYMENT",
        "labour": "EMPLOYMENT",
        "labor": "EMPLOYMENT",
        "cyber": "CYBER",
        "upi": "CYBER",
        "rti": "RTI",
        "right_to_information": "RTI",
        "contract": "CONTRACT",
        "property": "PROPERTY",
        "women": "WOMEN",
        "senior": "SENIOR",
        "motor_vehicle": "MOTOR_VEHICLE",
    }

    DOMAIN_KEYWORDS = {
        "TENANT": {
            "tenant",
            "landlord",
            "rent",
            "rental",
            "lease",
            "eviction",
            "security deposit",
            "deposit refund",
            "vacate",
        },
        "CONSUMER": {
            "consumer",
            "refund",
            "defective",
            "warranty",
            "replacement",
            "online shopping",
            "seller",
            "ecommerce",
            "e-commerce",
        },
        "EMPLOYMENT": {
            "employer",
            "employee",
            "salary",
            "wages",
            "notice period",
            "termination",
            "job",
            "workplace",
            "labour",
            "labor",
        },
        "CYBER": {
            "cyber",
            "upi",
            "phonepe",
            "google pay",
            "gpay",
            "paytm",
            "otp",
            "upi pin",
            "qr",
            "fraud",
            "scam",
            "tricked",
            "blackmail",
            "identity theft",
            "online threat",
            "unauthorized transaction",
            "sent money",
            "transferred money",
        },
        "RTI": {
            "rti",
            "right to information",
            "public authority",
            "government records",
            "information officer",
            "pio",
        },
        "CONTRACT": {
            "contract",
            "agreement",
            "clause",
            "liability",
            "indemnity",
            "arbitration",
            "auto renewal",
            "termination clause",
        },
        "PROPERTY": {"property", "ownership", "possession", "title deed", "land dispute"},
        "WOMEN": {"women", "woman", "harassment", "domestic violence", "stalking"},
        "SENIOR": {"senior citizen", "elder", "maintenance of parents"},
        "MOTOR_VEHICLE": {"motor vehicle", "driving licence", "driving license", "challan", "traffic fine"},
    }

    UNKNOWN_LEGAL_TOPICS = {
        "maritime",
        "aviation",
        "tax",
        "income tax",
        "patent",
        "trademark",
        "copyright",
        "criminal procedure",
        "family law",
        "divorce",
        "corporate law",
        "bankruptcy",
        "insolvency",
    }

    LEGAL_TERMS = {
        "law",
        "legal",
        "act",
        "rights",
        "right",
        "court",
        "case",
        "section",
        "complaint",
        "police",
        "lawyer",
        "contract",
        "agreement",
    }

    TITLE_REASON_MAP = {
        "Security Deposit Refund Rights": "tenant deposit refund and deduction guidance",
        "Tenant Eviction Rights": "tenant notice and eviction protection guidance",
        "Tenant Rights India": "general tenant rights background",
        "UPI Fraud Response Guide": "UPI fraud prevention and immediate action guidance",
        "Cybercrime Complaint Process": "cybercrime reporting and evidence steps",
        "Cyber Law India": "cyber law background and digital safety guidance",
        "Identity Theft Assistance Guide": "identity misuse and account-protection steps",
        "RTI Rights India": "RTI purpose, citizen access, and transparency rights",
        "Right to Information (RTI) Application Process": "RTI application and appeal procedure",
        "Online Blackmail Response Guide": "blackmail evidence and safety response steps",
        "Online Shopping Disputes": "consumer complaint and refund dispute guidance",
    }

    DISCLAIMER = (
        "This is legal information in plain language, not legal advice. "
        "For a decision about your exact situation, speak with a qualified lawyer."
    )

    def __init__(self) -> None:
        self.session_memory: Dict[str, Dict[str, Any]] = {}

    async def run(
        self,
        query: str,
        session_id: str,
        validation: ValidationResult,
        retriever: Any,
        llm_manager: Any = None,
        max_sources: int = 3,
    ) -> RetrievalPipelineResult:
        normalized_query = self._normalize_query(query)
        local_domain = self._detect_domain(normalized_query, validation)

        if self._should_stop_after_validation(validation, normalized_query, local_domain):
            return RetrievalPipelineResult(
                answer=validation.response or self._clarify_response(),
                citations=[],
                confidence=0.0,
                retrieved_matches=[],
            )

        if self._is_unknown_legal_topic(normalized_query, local_domain):
            return self._unknown_topic_response(normalized_query)

        if self._is_upi_act_query(normalized_query):
            return self._upi_act_response()

        intent = self._detect_intent(normalized_query, validation)
        rewritten_query = self._rewrite_with_memory(normalized_query, session_id, local_domain)
        expanded_query = self._expand_query(rewritten_query, local_domain, intent)

        matches = self._retrieve(retriever, expanded_query, local_domain)
        ranked_matches = self._rank_matches(matches, normalized_query, local_domain)
        relevant_matches = [m for m in ranked_matches if m.get("pipeline_score", 0.0) >= 0.18]

        if not relevant_matches:
            return self._unknown_topic_response(normalized_query)

        top_score = relevant_matches[0].get("pipeline_score", 0.0)
        if top_score < 0.27:
            return self._low_relevance_response(normalized_query, relevant_matches[:max_sources])

        selected = relevant_matches[:max_sources]
        confidence = self._confidence(top_score, selected, local_domain)
        citations = self._build_citations(selected)
        answer = self._format_answer(
            query=normalized_query,
            intent=intent,
            domain=local_domain,
            matches=selected,
            citations=citations,
            confidence=confidence,
        )
        self._remember(session_id, normalized_query, local_domain, intent)

        return RetrievalPipelineResult(
            answer=answer,
            citations=citations,
            confidence=confidence / 100,
            retrieved_matches=selected,
        )

    def _should_stop_after_validation(
        self,
        validation: ValidationResult,
        query: str,
        local_domain: Optional[str],
    ) -> bool:
        reason = (getattr(validation, "reason", "") or "").lower()
        action = getattr(validation, "action", None)

        if action == ValidationAction.CONTINUE:
            return False

        if action in {ValidationAction.GREETING, ValidationAction.GOODBYE, ValidationAction.REJECT}:
            return True

        if "incomplete" in reason or self._is_incomplete(query):
            return True

        if "non_legal" in reason or "non-legal" in reason:
            return True

        if "unsupported" in reason and not local_domain:
            return True

        if "low_confidence" in reason and not local_domain and not self._looks_legal(query):
            return True

        # Let the retrieval verifier make the final decision for legal-looking
        # questions that the validator could not classify confidently.
        return False

    def _looks_legal(self, query: str) -> bool:
        q = query.lower()
        return any(term in q for term in self.LEGAL_TERMS) or any(
            keyword in q
            for keywords in self.DOMAIN_KEYWORDS.values()
            for keyword in keywords
        )

    def _normalize_query(self, query: str) -> str:
        return re.sub(r"\s+", " ", (query or "").strip())

    def _is_incomplete(self, query: str) -> bool:
        q = query.lower().strip(" ?.")
        if len(q) < 4:
            return True
        incomplete_phrases = {
            "what is",
            "what are",
            "define",
            "explain",
            "tell me about",
            "how to",
            "how do i",
            "can i",
            "contract",
            "agreement",
            "rights",
            "law",
            "act",
        }
        return q in incomplete_phrases

    def _detect_domain(self, query: str, validation: ValidationResult) -> Optional[str]:
        detected = getattr(validation, "detected_domain", None)
        if detected:
            mapped = self.DOMAIN_ALIASES.get(str(detected).lower())
            if mapped:
                return mapped

        q = query.lower()
        scores: Dict[str, int] = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            scores[domain] = sum(1 for keyword in keywords if keyword in q)

        best_domain, best_score = max(scores.items(), key=lambda item: item[1])
        return best_domain if best_score > 0 else None

    def _detect_intent(self, query: str, validation: ValidationResult) -> str:
        q = query.lower().strip()
        if self._looks_like_title_query(q):
            return "information"

        if re.match(r"^(what is|what are|define|explain|tell me about|meaning of)\b", q):
            return "information"

        if re.match(r"^(how do i|how can i|how to|what should i do|what can i do|where can i)\b", q):
            return "guidance"

        guidance_terms = {
            "report",
            "complaint",
            "file a complaint",
            "scammed",
            "tricked",
            "lost money",
            "unauthorized transaction",
            "blackmail",
            "stolen",
            "threatening",
        }
        if any(term in q for term in guidance_terms):
            return "guidance"

        rights_terms = {
            "my landlord",
            "my employer",
            "refuses",
            "won't",
            "not returning",
            "withheld",
            "deducted",
            "evict",
            "fired",
            "salary",
            "deposit",
        }
        if any(term in q for term in rights_terms):
            return "rights"

        detected_intent = str(getattr(validation, "detected_intent", "") or "").lower()
        if any(word in detected_intent for word in ("definition", "explanation", "law")):
            return "information"
        if any(word in detected_intent for word in ("procedure", "remedy", "complaint")):
            return "guidance"
        if any(word in detected_intent for word in ("rights", "eligibility")):
            return "rights"

        return "information"

    def _looks_like_title_query(self, query: str) -> bool:
        words = query.split()
        if not 2 <= len(words) <= 6:
            return False
        action_words = {"how", "what", "why", "when", "where", "can", "should", "do", "report", "file"}
        return not any(word in action_words for word in words)

    def _rewrite_with_memory(self, query: str, session_id: str, domain: Optional[str]) -> str:
        memory = self.session_memory.get(session_id, {})
        q = query.lower()
        if memory and re.search(r"\b(it|this|that|one|they|them)\b", q):
            last_topic = memory.get("last_topic")
            last_domain = memory.get("last_domain")
            additions = []
            if last_topic:
                additions.append(f"previous topic: {last_topic}")
            if last_domain and not domain:
                additions.append(f"domain: {last_domain}")
            if additions:
                return f"{query} ({'; '.join(additions)})"
        return query

    def _expand_query(self, query: str, domain: Optional[str], intent: str) -> str:
        expansions = []
        if domain == "CYBER":
            expansions.extend(["cybercrime complaint", "UPI fraud", "unauthorized transaction", "save screenshots", "contact bank"])
        elif domain == "RTI":
            expansions.extend(["Right to Information Act", "public authority", "RTI application", "PIO", "appeal"])
        elif domain == "TENANT":
            expansions.extend(["rental agreement", "security deposit", "landlord", "notice", "tenant rights"])
            if any(term in query.lower() for term in ["law", "laws", "rights", "tenant rights", "tenant law"]):
                expansions.extend(["eviction rights", "maintenance obligations", "rent increase", "peaceful enjoyment"])
        elif domain == "EMPLOYMENT":
            expansions.extend(["salary", "notice period", "termination", "employment rights"])
        elif domain == "CONSUMER":
            expansions.extend(["refund", "defective product", "consumer complaint", "warranty"])
        elif domain == "CONTRACT":
            expansions.extend(["contract basics", "essential elements of contract", "offer", "acceptance", "consideration"])
            if any(term in query.lower() for term in ["risk", "risky", "red flag", "clause", "liability", "penalty", "indemnity"]):
                expansions.extend(["contract red flags", "liability clause", "penalty clause", "indemnity", "auto renewal"])

        if intent == "guidance":
            expansions.extend(["steps", "documents", "evidence", "complaint process"])
        elif intent == "rights":
            expansions.extend(["rights", "eligible", "documents", "legal protections"])

        return " ".join([query, *expansions])

    def _retrieve(self, retriever: Any, query: str, domain: Optional[str]) -> List[Dict[str, Any]]:
        if not retriever:
            return []

        domain_filter = domain
        try:
            filtered = retriever.retrieve(query, top_k=12, domain=domain_filter)
        except TypeError:
            filtered = retriever.retrieve(query, top_k=12)

        if filtered or domain:
            return filtered or []

        return retriever.retrieve(query, top_k=12)

    def _rank_matches(
        self,
        matches: Sequence[Dict[str, Any]],
        query: str,
        domain: Optional[str],
    ) -> List[Dict[str, Any]]:
        q_tokens = self._tokens(query)
        ranked = []
        for match in matches:
            metadata = match.get("metadata") or {}
            content = match.get("content") or match.get("text") or ""
            title = metadata.get("title") or self._extract_title(content)
            topics = metadata.get("topics") or metadata.get("category") or ""
            doc_domain = self._normalize_domain(metadata.get("domain") or metadata.get("category"))

            text_blob = f"{title} {topics} {content}".lower()
            overlap = len(q_tokens.intersection(self._tokens(text_blob)))
            overlap_score = min(overlap / max(len(q_tokens), 1), 1.0)

            base = float(match.get("combined_score") or match.get("score") or match.get("similarity") or 0.0)
            if base > 1:
                base = base / 100

            title_boost = 0.18 if any(token in self._tokens(str(title)) for token in q_tokens) else 0.0
            subtopic_boost = self._subtopic_boost(query.lower(), str(title).lower(), str(topics).lower(), content.lower())
            domain_boost = 0.18 if domain and doc_domain == domain else 0.0
            domain_penalty = -0.35 if domain and doc_domain and doc_domain != domain else 0.0

            pipeline_score = max(
                0.0,
                min(
                    1.0,
                    (base * 0.42)
                    + (overlap_score * 0.42)
                    + title_boost
                    + subtopic_boost
                    + domain_boost
                    + domain_penalty,
                ),
            )
            enriched = dict(match)
            enriched["pipeline_score"] = pipeline_score
            enriched["metadata"] = {**metadata, "title": title, "domain": doc_domain or metadata.get("domain")}
            ranked.append(enriched)

        return sorted(ranked, key=lambda item: item.get("pipeline_score", 0.0), reverse=True)

    def _subtopic_boost(self, query: str, title: str, topics: str, content: str) -> float:
        focused_text = f"{title} {topics}"
        full_text = f"{focused_text} {content[:500]}"
        boosts = 0.0

        phrase_pairs = [
            ("security deposit", ["security deposit", "deposit refund", "rental deposit"]),
            ("deposit", ["security deposit", "deposit refund", "rental deposit"]),
            ("upi", ["upi fraud", "cybercrime complaint"]),
            ("phonepe", ["upi fraud", "cybercrime complaint"]),
            ("rti", ["rti", "right to information"]),
            ("notice period", ["notice period", "employment contract"]),
            ("defective", ["defective product", "refund", "warranty"]),
            ("tenant rights", ["tenant rights", "eviction rights", "security deposit", "maintenance obligations"]),
            ("tenant laws", ["tenant rights", "eviction rights", "security deposit", "maintenance obligations"]),
        ]
        for query_phrase, source_phrases in phrase_pairs:
            if query_phrase in query:
                if any(phrase in focused_text for phrase in source_phrases):
                    boosts += 0.22
                elif any(phrase in full_text for phrase in source_phrases):
                    boosts += 0.08

        if "deposit" in query and "maintenance" in title:
            boosts -= 0.22
        if "tenant" in query and any(term in query for term in ["law", "laws", "rights"]):
            if any(phrase in focused_text for phrase in ["tenant rights", "eviction rights", "security deposit", "maintenance obligations", "rent increase"]):
                boosts += 0.28
        if "upi" in query and "blackmail" in title and "blackmail" not in query:
            boosts -= 0.12
        if "contract" in query:
            if any(term in query for term in ["risk", "risky", "red flag", "liability", "penalty", "indemnity", "auto renewal", "clause"]):
                if "red flags" in title or any(phrase in focused_text for phrase in ["liability", "penalty", "indemnity", "auto renewal"]):
                    boosts += 0.42
            if any(phrase in focused_text for phrase in ["contract basics", "essential elements", "types of contracts"]):
                boosts += 0.26
            if "red flags" in title and not any(term in query for term in ["risk", "risky", "red flag", "clause", "review"]):
                boosts -= 0.24

        return boosts

    def _tokens(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower()) if len(token) > 2}

    def _normalize_domain(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        upper = str(value).upper().replace(" ", "_")
        if "TENANT" in upper:
            return "TENANT"
        if "CONSUMER" in upper:
            return "CONSUMER"
        if "EMPLOY" in upper or "LABOUR" in upper or "LABOR" in upper:
            return "EMPLOYMENT"
        if "CYBER" in upper or "UPI" in upper:
            return "CYBER"
        if "RTI" in upper or "INFORMATION" in upper:
            return "RTI"
        if "CONTRACT" in upper:
            return "CONTRACT"
        if "PROPERTY" in upper:
            return "PROPERTY"
        if "WOMEN" in upper:
            return "WOMEN"
        if "SENIOR" in upper:
            return "SENIOR"
        if "MOTOR" in upper or "VEHICLE" in upper:
            return "MOTOR_VEHICLE"
        return upper

    def _is_unknown_legal_topic(self, query: str, domain: Optional[str]) -> bool:
        q = query.lower()
        if domain:
            return False
        if any(topic in q for topic in self.UNKNOWN_LEGAL_TOPICS):
            return True
        return any(term in q for term in self.LEGAL_TERMS)

    def _unknown_topic_response(self, query: str) -> RetrievalPipelineResult:
        answer = (
            "I am sorry, but I do not currently have enough verified information about this topic "
            "in my local legal knowledge base.\n\n"
            "Currently available areas include:\n\n"
            "- RTI and public information rights\n"
            "- Tenant and rental rights\n"
            "- Consumer complaints and refunds\n"
            "- Employment and notice period rights\n"
            "- Cyber law, UPI fraud, and online complaint steps\n"
            "- Contract clauses and risk review\n\n"
            "Please ask about one of these areas, or add verified legal documents for this topic to the knowledge base."
        )
        return RetrievalPipelineResult(answer=answer, citations=[], confidence=0.0, retrieved_matches=[])

    def _low_relevance_response(
        self,
        query: str,
        matches: Sequence[Dict[str, Any]],
    ) -> RetrievalPipelineResult:
        sources = self._build_citations(matches[:2])
        source_lines = "\n".join(f"- {source['title']}" for source in sources) or "- No strong source found"
        answer = (
            "I found a few possibly related legal notes, but they are not strong enough for a reliable answer.\n\n"
            "Please add a little more detail, such as the country, document type, issue, or the exact legal term you mean.\n\n"
            f"Closest sources checked:\n\n{source_lines}"
        )
        return RetrievalPipelineResult(answer=answer, citations=sources, confidence=0.42, retrieved_matches=list(matches))

    def _confidence(self, top_score: float, matches: Sequence[Dict[str, Any]], domain: Optional[str]) -> int:
        if top_score >= 0.78:
            confidence = 92
        elif top_score >= 0.58:
            confidence = 84
        elif top_score >= 0.42:
            confidence = 74
        else:
            confidence = 61

        if len(matches) >= 3:
            confidence += 2
        if domain:
            confidence += 1
        return max(55, min(confidence, 96))

    def _build_citations(self, matches: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        seen = set()
        for match in matches:
            metadata = match.get("metadata") or {}
            content = match.get("content") or match.get("text") or ""
            title = metadata.get("title") or self._extract_title(content)
            if not title or title in seen:
                continue
            seen.add(title)
            citations.append(
                {
                    "title": title,
                    "source": metadata.get("source") or metadata.get("source_file") or metadata.get("file_name") or title,
                    "excerpt": self._clean_excerpt(content),
                    "confidence": float(match.get("pipeline_score", 0.0)),
                    "reason": self.TITLE_REASON_MAP.get(title, self._reason_from_title(title)),
                }
            )
        return citations

    def _format_answer(
        self,
        query: str,
        intent: str,
        domain: Optional[str],
        matches: Sequence[Dict[str, Any]],
        citations: Sequence[Dict[str, Any]],
        confidence: int,
    ) -> str:
        if intent == "guidance":
            return self._format_guidance_answer(query, domain, matches, citations, confidence)
        if intent == "rights":
            return self._format_rights_answer(query, domain, matches, citations, confidence)
        return self._format_information_answer(query, domain, matches, citations, confidence)

    def _format_information_answer(
        self,
        query: str,
        domain: Optional[str],
        matches: Sequence[Dict[str, Any]],
        citations: Sequence[Dict[str, Any]],
        confidence: int,
    ) -> str:
        title = self._answer_title(query, domain, matches)
        explanation = self._short_explanation(query, domain, matches)
        key_points = self._key_points(domain, matches, mode="information")
        if domain == "TENANT" and self._is_tenant_rights_query(query):
            title = "Tenant Laws And Rights"
            explanation = (
                "Tenant laws generally protect tenants from arbitrary eviction, unfair deductions, unsafe living conditions, "
                "and sudden changes that violate the rental agreement or applicable local rules."
            )
            key_points = [
                "Tenants are generally entitled to proper notice before eviction or major tenancy changes.",
                "A landlord should not forcibly remove a tenant, lock the property, disconnect essential services, or remove belongings without legal process.",
                "Tenants may ask for a written explanation and evidence for any security deposit deductions.",
                "Landlords are generally expected to handle essential repairs and maintain a reasonably habitable property.",
                "Rent increase terms should usually follow the rental agreement and applicable local rules.",
            ]
        if domain == "CONTRACT" and self._is_contract_risk_query(query):
            title = "Contract Risk Clauses"
            explanation = (
                "Risky contract clauses are terms that can create unexpected cost, liability, lock-in, or dispute problems. "
                "They should be reviewed carefully before signing or accepting a revised agreement."
            )
            key_points = [
                "Watch for broad liability or indemnity clauses that make one side responsible for many losses.",
                "Check penalty, late fee, or damages clauses to see whether they are unusually high or unclear.",
                "Review auto-renewal, lock-in, and termination clauses before accepting the contract.",
                "Look for one-sided dispute resolution, unclear payment duties, and vague service obligations.",
            ]
        if domain == "CYBER" and self._is_cyber_document_query(query):
            title = "Documents For A Cyber Fraud Case"
            explanation = (
                "If you are preparing a cyber fraud complaint or court matter, the most useful documents are the ones "
                "that prove the transaction, the communication, the identity details available, and your reporting history."
            )
            key_points = [
                "Payment proof such as bank statement, UPI transaction ID, payment app receipt, or SMS confirmation.",
                "Screenshots of chats, calls, payment requests, QR codes, phone numbers, UPI IDs, usernames, and profile links.",
                "Complaint acknowledgements from the bank, payment app, cybercrime portal, police station, or other authority.",
                "A short timeline explaining when the fraud happened, how payment was requested, and what happened after payment.",
                "Any emails, notices, support tickets, account-freeze requests, or recovery communication.",
            ]
        if domain in {"CONTRACT", "RTI", "CYBER", "TENANT"}:
            return self._compose_information(
                title=title,
                intro=explanation,
                items=key_points,
                citations=citations,
                domain=domain,
                query=query,
            )
        remember = self._things_to_remember(domain)
        return self._compose(
            title=title,
            intro=explanation,
            primary_heading="Key Points",
            primary_items=key_points,
            secondary_heading="Things To Remember",
            secondary_items=remember,
            citations=citations,
            confidence=confidence,
        )

    def _compose_information(
        self,
        title: str,
        intro: str,
        items: Sequence[str],
        citations: Sequence[Dict[str, Any]],
        domain: Optional[str],
        query: str,
    ) -> str:
        item_lines = "\n".join(f"- {item}" for item in items)
        source_lines = "\n".join(f"- {c['title']}" for c in citations)
        example = self._example_for(domain, query)
        based_on = f"\n\nBased on:\n\n{source_lines}" if source_lines else ""
        follow_up = self._follow_up_for(domain, query)
        return (
            f"{title}\n\n"
            f"{intro}\n\n"
            f"In simple terms:\n\n{item_lines}"
            f"{example}"
            f"{based_on}\n\n"
            f"{self.DISCLAIMER}"
            f"{follow_up}"
        )

    def _example_for(self, domain: Optional[str], query: str) -> str:
        if domain == "CONTRACT" and not self._is_contract_risk_query(query):
            return (
                "\n\nExample:\n\n"
                "If Rahul agrees to paint Amit's house for a fixed price, and Amit accepts that offer, "
                "their agreement may become a contract if the legal requirements are satisfied."
            )
        if domain == "RTI":
            return (
                "\n\nExample:\n\n"
                "A citizen can file an RTI request asking a public authority for specific records, such as the status of an application or copies of public documents."
            )
        if domain == "CYBER" and "document" not in query.lower():
            return (
                "\n\nExample:\n\n"
                "If someone sends a fake payment link and tricks a user into approving a transaction, that may be treated as a cyber fraud complaint."
            )
        return ""

    def _follow_up_for(self, domain: Optional[str], query: str) -> str:
        if domain == "CONTRACT" and not self._is_contract_risk_query(query):
            return "\n\nYou can also ask me to explain the essential elements of a valid contract."
        if domain == "RTI":
            return "\n\nYou can also ask me how to file an RTI application or how the RTI appeal process works."
        if domain == "CYBER":
            return "\n\nIf this relates to a real incident, you can ask what immediate steps to take next."
        if domain == "TENANT":
            return "\n\nYou can also ask about eviction, security deposit refund, rent increase, or landlord maintenance duties."
        return ""

    def _is_contract_risk_query(self, query: str) -> bool:
        q = query.lower()
        return any(term in q for term in ["risk", "risky", "red flag", "liability", "penalty", "indemnity", "auto renewal", "clause"])

    def _is_tenant_rights_query(self, query: str) -> bool:
        q = query.lower()
        return "tenant" in q and any(term in q for term in ["law", "laws", "rights", "right"])

    def _format_guidance_answer(
        self,
        query: str,
        domain: Optional[str],
        matches: Sequence[Dict[str, Any]],
        citations: Sequence[Dict[str, Any]],
        confidence: int,
    ) -> str:
        title = self._guidance_title(domain, matches)
        intro = self._guidance_intro(domain)
        actions = self._key_points(domain, matches, mode="guidance")
        evidence = self._evidence_items(domain)
        return self._compose(
            title=title,
            intro=intro,
            primary_heading="Recommended Actions",
            primary_items=actions,
            secondary_heading="Evidence To Keep",
            secondary_items=evidence,
            citations=citations,
            confidence=confidence,
        )

    def _format_rights_answer(
        self,
        query: str,
        domain: Optional[str],
        matches: Sequence[Dict[str, Any]],
        citations: Sequence[Dict[str, Any]],
        confidence: int,
    ) -> str:
        title = self._rights_title(domain, matches)
        intro = self._rights_intro(domain)
        rights = self._key_points(domain, matches, mode="rights")
        next_steps = self._evidence_items(domain)
        return self._compose(
            title=title,
            intro=intro,
            primary_heading="Possible Rights",
            primary_items=rights,
            secondary_heading="Suggested Next Steps",
            secondary_items=next_steps,
            citations=citations,
            confidence=confidence,
        )

    def _compose(
        self,
        title: str,
        intro: str,
        primary_heading: str,
        primary_items: Sequence[str],
        secondary_heading: str,
        secondary_items: Sequence[str],
        citations: Sequence[Dict[str, Any]],
        confidence: int,
    ) -> str:
        primary = "\n".join(f"- {item}" for item in primary_items)
        secondary = "\n".join(f"- {item}" for item in secondary_items)
        source_lines = "\n".join(f"- {c['title']}" for c in citations)
        sources = f"\n\nBased on:\n\n{source_lines}" if source_lines else ""
        return (
            f"{title}\n\n"
            f"{intro}\n\n"
            f"{primary_heading}:\n\n{primary}\n\n"
            f"{secondary_heading}:\n\n{secondary}"
            f"{sources}\n\n"
            f"{self.DISCLAIMER}"
        )

    def _is_upi_act_query(self, query: str) -> bool:
        q = query.lower()
        return "upi" in q and "act" in q

    def _upi_act_response(self) -> RetrievalPipelineResult:
        answer = (
            "There is generally no separate law commonly called the \"UPI Act.\"\n\n"
            "UPI means Unified Payments Interface. It is a digital payment system, not a standalone Act by that name. "
            "Legal issues involving UPI usually connect to payment-system rules, cyber law, banking complaint processes, "
            "and fraud-reporting procedures.\n\n"
            "In simple terms:\n\n"
            "- If you mean UPI as a payment system, ask: \"What is UPI?\"\n"
            "- If you mean fraud through UPI, ask: \"How do I report UPI fraud?\"\n"
            "- If you mean the legal rules behind UPI, ask about cyber law, banking complaints, or digital payment disputes.\n\n"
            "The local knowledge base currently has stronger verified material on UPI fraud response and cybercrime complaints than on a separate \"UPI Act.\""
        )
        return RetrievalPipelineResult(answer=answer, citations=[], confidence=0.72, retrieved_matches=[])

    def _is_cyber_document_query(self, query: str) -> bool:
        q = query.lower()
        has_cyber = any(term in q for term in ["cyber", "upi fraud", "online fraud", "digital fraud"])
        has_docs = any(term in q for term in ["document", "documents", "submit", "submitted", "court", "evidence", "proof"])
        return has_cyber and has_docs

    def _answer_title(self, query: str, domain: Optional[str], matches: Sequence[Dict[str, Any]]) -> str:
        q = query.lower()
        if domain == "RTI":
            return "Right to Information (RTI) Act"
        if domain == "CYBER" and "upi" in q:
            return "UPI Fraud"
        if domain == "CYBER":
            return "Cyber Fraud"
        if domain == "CONTRACT":
            return "Contract Law"
        if matches:
            return str((matches[0].get("metadata") or {}).get("title") or "Legal Information")
        return "Legal Information"

    def _short_explanation(self, query: str, domain: Optional[str], matches: Sequence[Dict[str, Any]]) -> str:
        if domain == "RTI":
            return (
                "The RTI Act allows citizens to request information from public authorities. "
                "It is used to access government records, improve transparency, and make public bodies more accountable."
            )
        if domain == "CYBER" and "upi" in query.lower():
            return (
                "UPI fraud happens when someone tricks a person into approving a payment, sharing sensitive details, "
                "or transferring money through a digital payment app."
            )
        if domain == "CYBER":
            return (
                "Cyber fraud covers online cheating, identity misuse, payment scams, blackmail, and other digital offences. "
                "The safest first step is to preserve evidence and use official complaint channels."
            )
        if domain == "CONTRACT":
            return (
                "Contract law deals with legally enforceable agreements between two or more parties. "
                "A contract usually creates rights and obligations, such as payment duties, delivery duties, deadlines, remedies, and liability terms."
            )
        return self._summary_from_matches(matches)

    def _guidance_title(self, domain: Optional[str], matches: Sequence[Dict[str, Any]]) -> str:
        if domain == "CYBER":
            return "UPI / Cyber Fraud Reporting Steps"
        if domain == "RTI":
            return "RTI Application Steps"
        if domain == "TENANT":
            return "Tenant Issue Action Plan"
        return self._answer_title("", domain, matches)

    def _guidance_intro(self, domain: Optional[str]) -> str:
        if domain == "CYBER":
            return "Act quickly, avoid further payments, and preserve every digital record connected to the transaction."
        if domain == "RTI":
            return "An RTI request works best when it is specific, addressed to the right public authority, and supported by clear wording."
        if domain == "TENANT":
            return "Handle the dispute in writing and keep records so the issue can be reviewed later if needed."
        return "Use the verified sources below to take a careful, documented next step."

    def _rights_title(self, domain: Optional[str], matches: Sequence[Dict[str, Any]]) -> str:
        if domain == "TENANT":
            return "Tenant Rights"
        if domain == "EMPLOYMENT":
            return "Employment Rights"
        if domain == "CONSUMER":
            return "Consumer Rights"
        return "Possible Legal Rights"

    def _rights_intro(self, domain: Optional[str]) -> str:
        if domain == "TENANT":
            return "Based on the local tenant knowledge base, this issue may involve deposit records, written communication, and protection against unsupported deductions."
        if domain == "CYBER":
            return "This issue may involve cyber complaint steps and evidence preservation rather than a private negotiation alone."
        return "Based on the matching local legal sources, these are the most relevant rights or protections to consider."

    def _key_points(self, domain: Optional[str], matches: Sequence[Dict[str, Any]], mode: str) -> List[str]:
        if domain == "RTI":
            if mode == "guidance":
                return [
                    "Identify the public authority that holds the information.",
                    "Write a specific RTI request asking for records or information.",
                    "Submit the application through the accepted channel and keep the receipt.",
                    "Use the appeal process if the information is refused or delayed.",
                ]
            return [
                "Citizens can request information from public authorities.",
                "The purpose is transparency, accountability, and access to government records.",
                "Some information may be refused if a legal exemption applies.",
                "Appeals may be available if a response is delayed or denied.",
            ]

        if domain == "CYBER":
            if mode == "information":
                return [
                    "Cyber fraud can involve fake payment requests, phishing links, OTP misuse, identity theft, or online blackmail.",
                    "UPI fraud often depends on tricking the victim into approving a payment or sharing sensitive information.",
                    "Fast reporting and preserved evidence can improve the chance of action.",
                ]
            return [
                "Do not send more money or share OTP, PIN, password, or banking details.",
                "Contact your bank or payment app support immediately to report the transaction.",
                "File a cybercrime complaint through the appropriate official reporting channel.",
                "Keep the complaint number and follow up with the bank or authority.",
            ]

        if domain == "CONTRACT":
            if mode == "guidance":
                return [
                    "Read the agreement carefully before signing.",
                    "Check payment terms, deadlines, termination rights, liability, penalties, and renewal clauses.",
                    "Ask for unclear promises or duties to be written clearly in the contract.",
                    "Keep a signed copy and records of communication.",
                ]
            if mode == "rights":
                return [
                    "A party may rely on the written terms of a valid agreement.",
                    "A party may dispute unclear, unfair, or unsupported obligations depending on the facts and applicable law.",
                    "A party may seek remedies if the other side breaches important contract terms.",
                ]
            return [
                "A contract is a legally binding agreement that creates rights and obligations.",
                "Common elements include offer, acceptance, consideration, capacity, free consent, lawful object, and certainty of terms.",
                "Contracts may be written, oral, electronic, or implied depending on the situation and applicable law.",
                "Important clauses often cover payment, duration, termination, liability, indemnity, dispute resolution, and renewal.",
            ]

        if domain == "TENANT":
            return [
                "Request a written explanation for any security deposit deductions.",
                "Maintain deposit receipts, rent records, photos, and written communications.",
                "Dispute deductions that are not supported by the agreement or proof of damage.",
                "Review the rental agreement before accepting any deduction or eviction demand.",
            ]

        if domain == "EMPLOYMENT":
            return [
                "Check the offer letter, employment contract, and notice period terms.",
                "Keep salary slips, emails, attendance records, and written communications.",
                "Ask for written clarification if salary, termination, or notice terms are disputed.",
            ]

        if domain == "CONSUMER":
            return [
                "Preserve invoices, warranty cards, order details, and seller communication.",
                "Ask for repair, replacement, refund, or a written explanation depending on the issue.",
                "Escalate through the appropriate consumer complaint channel if the seller does not resolve it.",
            ]

        extracted = self._extract_bullets_from_matches(matches)
        return extracted[:4] or ["Use the matching legal sources below and keep written records before taking action."]

    def _evidence_items(self, domain: Optional[str]) -> List[str]:
        if domain == "CYBER":
            return [
                "Screenshots of chats, payment requests, transaction IDs, UPI IDs, phone numbers, and profile links.",
                "Bank or payment app confirmation messages and complaint numbers.",
                "A short timeline showing when the demand, payment, denial, or threat happened.",
            ]
        if domain == "TENANT":
            return [
                "Rental agreement, deposit receipt, rent receipts, photos, and move-in or move-out records.",
                "Written messages requesting the refund or asking for itemized deductions.",
                "Any notice, inspection report, or communication from the landlord.",
            ]
        return [
            "Receipts, notices, emails, screenshots, agreements, and complaint acknowledgements.",
            "A timeline of what happened and who said what.",
            "Copies of documents mentioned in the matching sources.",
        ]

    def _things_to_remember(self, domain: Optional[str]) -> List[str]:
        if domain == "RTI":
            return [
                "Ask for specific records, not broad explanations.",
                "Use the correct public authority whenever possible.",
                "Keep proof of submission and response dates.",
            ]
        if domain == "CYBER":
            return [
                "Do not negotiate with scammers or reveal sensitive details.",
                "Use official bank, payment app, and cybercrime reporting channels.",
                "Preserve evidence before deleting chats or blocking accounts.",
            ]
        if domain == "TENANT":
            return [
                "Keep the rental agreement, rent receipts, deposit proof, notices, repair requests, photos, and written messages.",
                "Communicate important tenancy issues in writing whenever possible.",
                "Use appropriate complaint or legal processes if informal resolution fails.",
            ]
        if domain == "CONTRACT":
            return [
                "Do not rely only on verbal promises if the written contract says something different.",
                "Unclear payment, liability, penalty, renewal, or termination clauses should be reviewed carefully.",
                "Keep signed copies, emails, invoices, notices, and proof of performance.",
            ]
        return [
            "Keep written records and supporting documents.",
            "Check the relevant agreement or policy before acting.",
            "Use official complaint or legal channels when informal resolution fails.",
        ]

    def _summary_from_matches(self, matches: Sequence[Dict[str, Any]]) -> str:
        if not matches:
            return "I found a matching legal topic in the local knowledge base."
        text = matches[0].get("content") or matches[0].get("text") or ""
        overview = self._section_text(text, "OVERVIEW")
        if overview:
            sentences = re.split(r"(?<=[.!?])\s+", overview.strip())
            return " ".join(sentences[:2])
        return self._clean_excerpt(text, 260)

    def _extract_bullets_from_matches(self, matches: Sequence[Dict[str, Any]]) -> List[str]:
        bullets: List[str] = []
        for match in matches:
            text = match.get("content") or match.get("text") or ""
            for line in text.splitlines():
                clean = line.strip(" -\t")
                if len(clean) < 12:
                    continue
                if re.match(r"^\d+\.", clean):
                    clean = re.sub(r"^\d+\.\s*", "", clean)
                if clean.endswith(":") or clean.upper() == clean:
                    continue
                if clean not in bullets:
                    bullets.append(clean)
                if len(bullets) >= 5:
                    return bullets
        return bullets

    def _section_text(self, text: str, section: str) -> str:
        pattern = re.compile(rf"{section}:\s*(.*?)(?:\n[A-Z][A-Z\s/]+:|\Z)", re.DOTALL)
        match = pattern.search(text)
        return match.group(1).strip() if match else ""

    def _extract_title(self, content: str) -> str:
        match = re.search(r"TITLE:\s*(.+)", content or "", re.IGNORECASE)
        if match:
            return match.group(1).strip()
        first = (content or "").strip().splitlines()
        return first[0][:80] if first else "Legal Source"

    def _clean_excerpt(self, content: str, limit: int = 320) -> str:
        excerpt = (content or "").replace("\ufeff", "").replace("ï»¿", "")
        excerpt = excerpt.replace("Ã¢â\x82¬Â¢", "-").replace("â€¢", "-")
        excerpt = re.sub(r"\s+", " ", excerpt.strip())
        return excerpt[:limit].rstrip()

    def _reason_from_title(self, title: str) -> str:
        words = re.sub(r"[^A-Za-z0-9 ]+", "", title).strip().lower()
        return f"matched local knowledge about {words}" if words else "matched local legal knowledge"

    def _remember(self, session_id: str, query: str, domain: Optional[str], intent: str) -> None:
        self.session_memory[session_id] = {
            "last_topic": query,
            "last_domain": domain,
            "last_intent": intent,
        }

    def _clarify_response(self) -> str:
        return (
            "Could you please complete your question?\n\n"
            "For example:\n\n"
            "- What is the RTI Act?\n"
            "- How do I report UPI fraud?\n"
            "- My landlord refuses to return my security deposit. What should I do?"
        )
