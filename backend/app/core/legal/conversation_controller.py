from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from app.core.legal.query_validator import QueryValidator
from app.core.legal.retrieval_pipeline import RetrievalPipeline, RetrievalPipelineResult
from app.db.sqlite import SQLiteRepository


class ConversationIntent(str, Enum):
    GREETING = "greeting"
    GOODBYE = "goodbye"
    THANK_YOU = "thank_you"
    HELP = "help"
    IDENTITY = "identity"
    CREATOR = "creator"
    CAPABILITY = "capability"
    LIMITATIONS = "limitations"
    PURPOSE = "purpose"
    ERROR_RECOVERY = "error_recovery"
    SMALL_TALK = "small_talk"
    NON_LEGAL = "non_legal"
    RESET = "reset"
    LEGAL_QUERY = "legal_query"


@dataclass
class ConversationSession:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    greeting_done: bool = False
    conversation_stage: str = "INTRO"
    non_legal_count: int = 0
    knowledge_gap_counter: Dict[str, int] = field(default_factory=dict)
    repeated_topics: Dict[str, int] = field(default_factory=dict)
    last_unknown_topic: Optional[str] = None
    last_domain: Optional[str] = None
    last_topic: Optional[str] = None
    last_answer: Optional[str] = None
    messages: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ControllerResult:
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    intent: str = ConversationIntent.LEGAL_QUERY.value
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationController:
    """State machine for user-facing chat behavior.

    The controller owns conversational decisions such as greeting, help,
    non-legal refusals, repeated knowledge gaps, and reset. Legal retrieval
    remains delegated to QueryValidator + RetrievalPipeline.
    """

    LEGAL_SCOPE = (
        "RTI, tenant rights, consumer rights, employment rights, cyber law and UPI fraud, "
        "contracts, property, women rights, senior citizen rights, and motor vehicle topics"
    )

    GREETINGS = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "namaste",
    }
    GOODBYES = {"bye", "goodbye", "see you", "see you later", "exit", "quit"}
    THANKS = {"thanks", "thank you", "thx", "thankyou"}
    HELP = {"help", "what can you do", "what can i ask", "show help", "topics"}
    IDENTITY = {
        "what are you",
        "who are you",
        "what is this assistant",
        "what kind of assistant are you",
        "are you an ai",
    }
    CREATOR = {
        "who made you",
        "who created you",
        "who developed you",
        "who built you",
        "who is your creator",
        "who is your developer",
        "who is the owner of you",
        "who owns you",
        "who is your owner",
    }
    CAPABILITY = {
        "how do you work",
        "how you work",
        "how can you answer legal questions",
        "how you can give me legal question's answers",
        "how can you give me legal answers",
        "how do you answer legal questions",
        "how do you give legal answers",
        "how can you help me",
    }
    LIMITATIONS = {
        "what are your weaknesses",
        "what are the weaknesses of you",
        "what can you not do",
        "what are your limitations",
        "what are your limits",
    }
    PURPOSE = {
        "why were you built",
        "what is your purpose",
        "why do you exist",
        "why are you made",
    }
    ERROR_RECOVERY = {
        "what wrong with you",
        "what is wrong with you",
        "what's wrong with you",
        "you are wrong",
        "that is wrong",
    }
    SMALL_TALK = {
        "how are you",
        "how are you?",
        "are you there",
    }
    RESET = {"reset", "clear chat", "start over", "new chat"}

    NON_LEGAL_KEYWORDS = {
        "pizza",
        "recipe",
        "cricket",
        "ipl",
        "football",
        "movie",
        "song",
        "weather",
        "time",
        "current time",
        "today",
        "winner",
        "result",
        "python code",
        "javascript",
        "programming",
        "write code",
        "machine learning",
        "engineering",
    }

    UNKNOWN_TEMPLATES = [
        "I do not currently have verified legal information on this topic in my local knowledge base.",
        "This legal topic is recognized, but it has not yet been covered well enough by verified local documents.",
        "I cannot answer this accurately because my verified legal documents do not yet cover this subject.",
        "The administrator has not yet added reliable legal resources for this topic.",
    ]

    def __init__(
        self,
        validator: Optional[QueryValidator] = None,
        retrieval_pipeline: Optional[RetrievalPipeline] = None,
    ) -> None:
        self.validator = validator or QueryValidator()
        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline()
        self.sessions: Dict[str, ConversationSession] = {}

    def get_session(self, session_id: str) -> ConversationSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id=session_id)
        session = self.sessions[session_id]
        session.last_active = time.time()
        return session

    async def process_message(
        self,
        *,
        session_id: str,
        message: str,
        repository: SQLiteRepository,
        retriever: Any,
        llm_manager: Any = None,
    ) -> ControllerResult:
        session = self.get_session(session_id)
        text = self._normalize(message)

        repository.add_chat_message(session_id, "user", message)
        session.messages.append({"role": "user", "content": message})

        direct_intent = self._detect_direct_intent(text)
        if direct_intent:
            result = self._handle_direct_intent(direct_intent, text, session)
            self._store_assistant(repository, session, result.answer)
            return result

        validation = self.validator.validate(text, session_id=session_id)
        pipeline_result = await self.retrieval_pipeline.run(
            query=text,
            session_id=session_id,
            validation=validation,
            retriever=retriever,
            llm_manager=llm_manager,
        )

        result = self._post_process_pipeline_result(text, session, pipeline_result)
        self._store_assistant(repository, session, result.answer)
        return result

    def _detect_direct_intent(self, text: str) -> Optional[ConversationIntent]:
        q = text.lower().strip(" .!?")
        if not q:
            return ConversationIntent.HELP
        if q in self.GREETINGS:
            return ConversationIntent.GREETING
        if q in self.GOODBYES:
            return ConversationIntent.GOODBYE
        if q in self.THANKS:
            return ConversationIntent.THANK_YOU
        if q in self.HELP:
            return ConversationIntent.HELP
        if q in self.CREATOR or self._looks_like_creator_question(q):
            return ConversationIntent.CREATOR
        if q in self.IDENTITY:
            return ConversationIntent.IDENTITY
        if q in self.CAPABILITY or self._looks_like_capability_question(q):
            return ConversationIntent.CAPABILITY
        if q in self.LIMITATIONS:
            return ConversationIntent.LIMITATIONS
        if q in self.PURPOSE:
            return ConversationIntent.PURPOSE
        if q in self.ERROR_RECOVERY:
            return ConversationIntent.ERROR_RECOVERY
        if q in self.SMALL_TALK:
            return ConversationIntent.SMALL_TALK
        if q in self.RESET:
            return ConversationIntent.RESET
        if self._looks_like_math_question(q):
            return ConversationIntent.NON_LEGAL
        if self._looks_like_current_affairs_or_time(q):
            return ConversationIntent.NON_LEGAL
        if any(keyword in q for keyword in self.NON_LEGAL_KEYWORDS) and not self._looks_legal(q):
            return ConversationIntent.NON_LEGAL
        return None

    def _handle_direct_intent(
        self,
        intent: ConversationIntent,
        text: str,
        session: ConversationSession,
    ) -> ControllerResult:
        if intent == ConversationIntent.GREETING:
            if session.greeting_done:
                answer = (
                    "Welcome back. What legal question would you like to discuss next?\n\n"
                    f"I can help with {self.LEGAL_SCOPE}."
                )
            else:
                session.greeting_done = True
                session.conversation_stage = "DISCUSSION"
                answer = (
                    "Hello!\n\n"
                    "I'm your Legal Knowledge Assistant.\n\n"
                    "I can help explain legal rights, contracts, RTI, cyber law, consumer protection, "
                    "tenancy, employment law, and government procedures in simple language.\n\n"
                    "What legal question can I help you with today?"
                )
            return ControllerResult(answer=answer, intent=intent.value, confidence=1.0)

        if intent == ConversationIntent.GOODBYE:
            session.conversation_stage = "ENDING"
            answer = (
                "Thank you for using the Legal Knowledge Assistant.\n\n"
                "If you need legal information or want to understand your legal rights in the future, "
                "feel free to return.\n\n"
                "Have a wonderful day!"
            )
            return ControllerResult(answer=answer, intent=intent.value, confidence=1.0)

        if intent == ConversationIntent.THANK_YOU:
            answer = (
                "You're welcome. I'm here whenever you want to understand a legal right, "
                "contract clause, government process, or document in simpler language."
            )
            return ControllerResult(answer=answer, intent=intent.value, confidence=1.0)

        if intent == ConversationIntent.HELP:
            answer = (
                "I can currently help with:\n\n"
                "- RTI and public information requests\n"
                "- Tenant rights and rental disputes\n"
                "- Consumer rights, refunds, and defective products\n"
                "- Employment rights and notice periods\n"
                "- Cyber law, UPI fraud, and cybercrime complaints\n"
                "- Contract clauses, risks, and document simplification\n"
                "- Property, women rights, senior citizen rights, and motor vehicle topics\n\n"
                "Example questions:\n\n"
                "- What is the RTI Act?\n"
                "- How do I report UPI fraud?\n"
                "- My landlord refuses to return my security deposit. What should I do?\n"
                "- What are risky clauses in this contract?"
            )
            return ControllerResult(answer=answer, intent=intent.value, confidence=1.0)

        if intent == ConversationIntent.CREATOR:
            answer = (
                "I was made by Rahul aka Saikat Roy as a Legal Knowledge Assistant.\n\n"
                "My purpose is to help users understand legal rights, contracts, government procedures, "
                "and legal documents using a verified local knowledge base."
            )
            return ControllerResult(answer=answer, intent=intent.value, confidence=1.0)

        if intent == ConversationIntent.IDENTITY:
            answer = (
                "I'm a Legal Knowledge Assistant built for this project.\n\n"
                "I help explain legal rights, contracts, RTI, cyber law, consumer issues, tenancy, "
                "employment topics, and related legal documents in simple language. I stay grounded in the "
                "local legal knowledge base and avoid making up legal facts."
            )
            return ControllerResult(answer=answer, intent=intent.value, confidence=1.0)

        if intent == ConversationIntent.CAPABILITY:
            answer = (
                "I answer legal questions by searching the verified local legal knowledge base first.\n\n"
                "My process is:\n\n"
                "- Understand the question and legal domain.\n"
                "- Search relevant local legal documents.\n"
                "- Check whether the retrieved sources are strong enough.\n"
                "- Format the answer in plain language with sources and a legal-information disclaimer.\n\n"
                "If the knowledge base does not contain enough verified information, I should say so instead of guessing."
            )
            return ControllerResult(answer=answer, intent=intent.value, confidence=1.0)

        if intent == ConversationIntent.LIMITATIONS:
            answer = (
                "My main limitations are:\n\n"
                "- I can only answer reliably from the legal knowledge base available in this project.\n"
                "- I provide legal information, not personal legal advice.\n"
                "- I may not cover legal topics that have not been added to the local database yet.\n"
                "- I should not answer unrelated topics such as sports, cooking, entertainment, or programming.\n\n"
                "For a real legal decision, a qualified lawyer should review the exact facts and documents."
            )
            return ControllerResult(answer=answer, intent=intent.value, confidence=1.0)

        if intent == ConversationIntent.PURPOSE:
            answer = (
                "I was built to make legal information easier to understand.\n\n"
                "The goal is to help users simplify legal documents, understand rights, identify contract risks, "
                "compare document versions, and find relevant sources from the local legal knowledge base."
            )
            return ControllerResult(answer=answer, intent=intent.value, confidence=1.0)

        if intent == ConversationIntent.ERROR_RECOVERY:
            answer = (
                "I may have misunderstood your previous question.\n\n"
                "Tell me what you meant, or rephrase the legal issue a little, and I'll try to answer it more clearly."
            )
            return ControllerResult(answer=answer, intent=intent.value, confidence=1.0)

        if intent == ConversationIntent.SMALL_TALK:
            answer = (
                "I'm doing well, thank you. I'm ready to help answer legal questions "
                "and explain legal rights in simple language."
            )
            return ControllerResult(answer=answer, intent=intent.value, confidence=1.0)

        if intent == ConversationIntent.RESET:
            session.messages.clear()
            session.knowledge_gap_counter.clear()
            session.repeated_topics.clear()
            session.last_unknown_topic = None
            session.last_domain = None
            session.last_topic = None
            session.last_answer = None
            session.conversation_stage = "INTRO"
            answer = "Chat context has been reset. What legal question would you like to ask now?"
            return ControllerResult(answer=answer, intent=intent.value, confidence=1.0)

        session.non_legal_count += 1
        if session.non_legal_count >= 3:
            answer = (
                "I'm here specifically to assist with legal rights, legal information, contracts, "
                "and government procedures. If you do not need legal assistance, this chat may not be "
                "the right tool. Please keep the conversation focused on legal or rights-related topics."
            )
        else:
            answer = (
                "I'm designed specifically to answer questions related to law, legal rights, contracts, "
                "and government procedures.\n\n"
                "I can't reliably answer questions outside those areas. Please ask a law-related question."
            )
        return ControllerResult(answer=answer, intent=intent.value, confidence=1.0)

    def _post_process_pipeline_result(
        self,
        text: str,
        session: ConversationSession,
        pipeline_result: RetrievalPipelineResult,
    ) -> ControllerResult:
        answer = pipeline_result.answer
        intent = ConversationIntent.LEGAL_QUERY.value

        if self._is_knowledge_gap(answer):
            topic = self._topic_key(text)
            session.knowledge_gap_counter[topic] = session.knowledge_gap_counter.get(topic, 0) + 1
            session.last_unknown_topic = topic
            answer = self._knowledge_gap_response(topic, session.knowledge_gap_counter[topic])
            intent = "knowledge_gap"
        else:
            session.last_topic = text
            session.last_answer = answer
            session.last_domain = self._infer_domain_from_answer(answer)

        return ControllerResult(
            answer=answer,
            citations=pipeline_result.citations,
            confidence=pipeline_result.confidence,
            intent=intent,
            metadata={"retrieved_matches": len(pipeline_result.retrieved_matches)},
        )

    def _knowledge_gap_response(self, topic: str, count: int) -> str:
        topic_label = topic.replace("_", " ").title()
        if count == 1:
            opening = self.UNKNOWN_TEMPLATES[0]
        elif count == 2:
            opening = self.UNKNOWN_TEMPLATES[1]
        elif count == 3:
            opening = (
                f"It looks like you've asked about {topic_label} a few times. "
                "This section may still be waiting for verified knowledge-base documents."
            )
        else:
            opening = (
                f"I still do not have enough verified local material for {topic_label}. "
                "The knowledge base needs to be expanded before I can answer this reliably."
            )

        return (
            f"{opening}\n\n"
            "Currently available areas include:\n\n"
            "- RTI and public information rights\n"
            "- Tenant and rental rights\n"
            "- Consumer complaints and refunds\n"
            "- Employment and notice period rights\n"
            "- Cyber law, UPI fraud, and online complaint steps\n"
            "- Contract clauses and risk review\n\n"
            "You can add verified legal documents for this topic and rebuild the knowledge base index."
        )

    def _store_assistant(
        self,
        repository: SQLiteRepository,
        session: ConversationSession,
        answer: str,
    ) -> None:
        repository.add_chat_message(session.session_id, "assistant", answer)
        session.messages.append({"role": "assistant", "content": answer})

    def _normalize(self, message: str) -> str:
        return re.sub(r"\s+", " ", (message or "").strip())

    def _looks_like_creator_question(self, text: str) -> bool:
        return (
            any(word in text for word in ["made", "created", "developed", "built", "owner", "owns"]) and any(
            target in text for target in ["you", "this bot", "assistant", "chatbot"]
        )
        )

    def _looks_like_capability_question(self, text: str) -> bool:
        has_answer_word = any(word in text for word in ["answer", "give", "provide", "work", "search"])
        has_legal_word = any(word in text for word in ["legal", "law", "rights", "question", "questions"])
        return has_answer_word and has_legal_word

    def _looks_like_math_question(self, text: str) -> bool:
        return bool(re.search(r"\b\d+\s*[\+\-\*/x]\s*\d+\b", text)) or text in {"math", "calculation"}

    def _looks_like_current_affairs_or_time(self, text: str) -> bool:
        legal_markers = [
            "legal",
            "law",
            "court",
            "rti",
            "contract",
            "tenant",
            "consumer",
            "employment",
            "cyber",
            "fraud",
            "rights",
            "act",
        ]
        if any(marker in text for marker in legal_markers):
            return False
        current_terms = ["current time", "time in", "today's", "todays", "today", "this year", "winner", "result"]
        non_legal_domains = ["ipl", "cricket", "football", "weather", "movie", "election result"]
        return any(term in text for term in current_terms) or any(term in text for term in non_legal_domains)

    def _looks_legal(self, text: str) -> bool:
        legal_terms = {
            "law",
            "legal",
            "rights",
            "right",
            "act",
            "court",
            "contract",
            "agreement",
            "tenant",
            "landlord",
            "consumer",
            "rti",
            "cyber",
            "fraud",
            "salary",
            "employment",
            "complaint",
            "police",
            "property",
        }
        return any(term in text for term in legal_terms)

    def _is_knowledge_gap(self, answer: str) -> bool:
        lowered = answer.lower()
        return (
            "do not currently have enough verified information" in lowered
            or "closest sources checked" in lowered
            or "not strong enough for a reliable answer" in lowered
        )

    def _topic_key(self, text: str) -> str:
        q = text.lower()
        known_topics = [
            "contract",
            "maritime",
            "family",
            "divorce",
            "criminal",
            "tax",
            "copyright",
            "trademark",
            "patent",
            "aviation",
            "corporate",
        ]
        for topic in known_topics:
            if topic in q:
                return topic.replace(" ", "_")
        tokens = [token for token in re.findall(r"[a-zA-Z]+", q) if len(token) > 3]
        return "_".join(tokens[:3]) if tokens else "unknown_topic"

    def _infer_domain_from_answer(self, answer: str) -> Optional[str]:
        lowered = answer.lower()
        for domain in ["tenant", "consumer", "employment", "cyber", "rti", "contract", "property"]:
            if domain in lowered:
                return domain
        return None
