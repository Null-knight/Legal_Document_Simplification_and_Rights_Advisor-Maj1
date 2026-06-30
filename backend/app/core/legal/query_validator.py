from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from abc import ABC, abstractmethod


# ──────────────────────────────────────────────────────────────
# ENUMS AND DATA CLASSES
# ──────────────────────────────────────────────────────────────

class ValidationAction(Enum):
    CONTINUE = "continue"
    CLARIFY = "clarify"
    REJECT = "reject"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


class QueryIntent(Enum):
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    PROCEDURE = "procedure"
    RIGHTS = "rights"
    COMPLAINT = "complaint"
    COMPARISON = "comparison"
    PENALTY = "penalty"
    PUNISHMENT = "punishment"
    APPEAL = "appeal"
    DOCUMENT = "document"
    EXAMPLES = "examples"
    CHECKLIST = "checklist"
    ELIGIBILITY = "eligibility"
    TIME_LIMIT = "time_limit"
    AUTHORITY = "authority"
    COURT = "court"
    SECTION = "section"
    FINE = "fine"
    FORMS = "forms"
    DRAFTING = "drafting"
    CASE_ANALYSIS = "case_analysis"
    DECISION_SUPPORT = "decision_support"
    SUMMARIZATION = "summarization"
    DOCUMENT_SIMPLIFICATION = "document_simplification"
    LEGAL_DRAFT_REVIEW = "legal_draft_review"
    RIGHTS_IDENTIFICATION = "rights_identification"
    RISK_ANALYSIS = "risk_analysis"
    LAW_EXPLANATION = "law_explanation"
    COMPLAINT_DRAFTING = "complaint_drafting"
    LETTER_DRAFTING = "letter_drafting"
    NOTICE_DRAFTING = "notice_drafting"
    GENERAL = "general"
    UNKNOWN = "unknown"


class QueryComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"


class RetrievalStrategy(Enum):
    NO_RETRIEVAL = "no_retrieval"
    DIRECT_KB = "direct_kb"
    RAG = "rag"
    MULTI_DOCUMENT = "multi_document"
    HYBRID_SEARCH = "hybrid_search"
    WEB_SEARCH = "web_search"


class EmotionalState(Enum):
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    CONFUSED = "confused"
    POLITE = "polite"
    URGENT = "urgent"
    SATISFIED = "satisfied"


class ResponseStyle(Enum):
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    RIGHTS = "rights"
    COMPLAINT = "complaint"
    COMPARISON = "comparison"
    PENALTY = "penalty"
    EXAMPLES = "examples"
    CHECKLIST = "checklist"
    ELIGIBILITY = "eligibility"
    APPEAL = "appeal"
    CASE_ANALYSIS = "case_analysis"
    DECISION_SUPPORT = "decision_support"
    SUMMARIZATION = "summarization"
    DRAFTING = "drafting"
    GENERAL = "general"


@dataclass
class ExtractedEntities:
    acts: List[str] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)
    courts: List[str] = field(default_factory=list)
    authorities: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    time_periods: List[str] = field(default_factory=list)
    penalties: List[str] = field(default_factory=list)
    rights: List[str] = field(default_factory=list)
    forms: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    amounts: List[str] = field(default_factory=list)
    legal_concepts: List[str] = field(default_factory=list)  # New: Offer, Acceptance, etc.


@dataclass
class ConfidenceScores:
    input_quality: float = 0.0
    domain_confidence: float = 0.0
    intent_confidence: float = 0.0
    knowledge_coverage: float = 0.0
    conversation_context: float = 0.0
    retrieval_score: float = 0.0
    embedding_similarity: float = 0.0
    metadata_score: float = 0.0
    completeness_score: float = 0.0  # New: query completeness
    final_confidence: float = 0.0


@dataclass
class ValidationResult:
    valid: bool
    action: ValidationAction
    normalized_query: str = ""
    reason: str = ""
    response: str = ""
    confidence: ConfidenceScores = field(default_factory=ConfidenceScores)
    detected_domain: Optional[str] = None
    domain_scores: Dict[str, float] = field(default_factory=dict)
    domain_hierarchy: List[Tuple[str, float]] = field(default_factory=list)  # New: prioritized domains
    detected_intent: QueryIntent = QueryIntent.UNKNOWN
    intent_scores: Dict[str, float] = field(default_factory=dict)
    intent_hierarchy: List[Tuple[str, float]] = field(default_factory=list)  # New: prioritized intents
    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    is_followup: bool = False
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.RAG
    kb_has_knowledge: bool = True
    entities: ExtractedEntities = field(default_factory=ExtractedEntities)
    query_complexity: QueryComplexity = QueryComplexity.SIMPLE
    suggested_topics: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    logs: List[Dict[str, Any]] = field(default_factory=list)
    corrected_query: Optional[str] = None
    query_rewrite: Optional[str] = None  # NEW: Cleaned query for retriever
    retrieval_keywords: List[str] = field(default_factory=list)  # NEW: Keywords for retriever
    response_style: Optional[ResponseStyle] = None  # NEW: Response template to use
    missing_knowledge: List[str] = field(default_factory=list)  # NEW: Detected knowledge gaps
    primary_intent: QueryIntent = QueryIntent.UNKNOWN
    secondary_intent: Optional[QueryIntent] = None
    tertiary_intent: Optional[QueryIntent] = None
    response_template: Optional[str] = None  # NEW: Structured template for response


@dataclass
class SessionState:
    session_id: str
    unrelated_questions: int = 0
    clarification_count: int = 0
    last_domain: Optional[str] = None
    last_domain_scores: Dict[str, float] = field(default_factory=dict)
    last_intent: Optional[QueryIntent] = None
    last_question: str = ""
    last_question_normalized: str = ""
    last_response: str = ""
    question_count: int = 0
    followup_count: int = 0
    topic_history: List[str] = field(default_factory=list)
    entity_history: List[ExtractedEntities] = field(default_factory=list)
    cited_laws: List[str] = field(default_factory=list)
    cited_sections: List[str] = field(default_factory=list)
    last_retrieved_docs: List[str] = field(default_factory=list)
    frustration_count: int = 0
    satisfaction_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    question_sequence: List[str] = field(default_factory=list)
    # New fields for follow-up resolution
    last_domain_context: str = ""
    last_entity_context: str = ""
    last_topic_context: str = ""


# ──────────────────────────────────────────────────────────────
# INCOMPLETE QUERY DETECTOR (NEW MODULE)
# ──────────────────────────────────────────────────────────────

class IncompleteQueryDetector:
    """Detects incomplete queries that need clarification."""
    
    def __init__(self):
        self.question_words = {"what", "who", "when", "where", "why", "how", "which", "whose"}
        self.legal_nouns = {
            "act", "law", "section", "clause", "rule", "regulation", "court", "tribunal",
            "authority", "commission", "board", "procedure", "right", "complaint", "appeal"
        }
        self.incomplete_patterns = [
            (r"^what\s*$", "What would you like to know?"),
            (r"^how\s*$", "How can I help you?"),
            (r"^why\s*$", "Why what? Please elaborate."),
            (r"^can i\s*$", "Can you complete your question?"),
            (r"^is it\s*$", "Is it what? Please specify."),
            (r"^tell me\s*$", "Tell me about what?"),
            (r"^explain\s*$", "Explain what? Please provide more details."),
        ]
    
    def detect(self, text: str) -> Tuple[bool, str, float]:
        """Detect if query is incomplete and return reason and confidence."""
        text_lower = text.lower().strip()
        words = re.findall(r"\b\w+\b", text_lower)
        
        # Check for complete sentence structure
        if self._is_complete_sentence(text_lower, words):
            return False, "", 0.0
        
        # Check for incomplete patterns
        for pattern, _ in self.incomplete_patterns:
            if re.match(pattern, text_lower):
                return True, "incomplete_pattern", 0.9
        
        # Check for single word queries
        if len(words) == 1:
            return True, "single_word", 0.8
        
        # Check for question without subject
        if self._starts_with_question_word(words) and len(words) < 3:
            return True, "question_without_subject", 0.7
        
        # Check for missing legal context
        if self._missing_legal_context(words):
            return True, "missing_legal_context", 0.6
        
        # Check for pronouns without reference
        if self._has_pronoun_without_reference(words):
            return True, "pronoun_without_reference", 0.7
        
        return False, "", 0.0
    
    def _is_complete_sentence(self, text: str, words: List[str]) -> bool:
        """Check if text is a complete sentence."""
        if len(words) < 3:
            return False
        
        # Check for subject-verb structure (simplified)
        has_subject = any(word in text for word in {"i", "you", "he", "she", "it", "we", "they"})
        has_verb = any(word in text for word in {"is", "are", "was", "were", "have", "has", "do", "does"})
        
        if has_subject and has_verb:
            return True
        
        # Check for imperative structure
        if text.startswith(("please", "tell", "show", "give", "provide", "explain")):
            return True
        
        # Check for question structure
        if self._starts_with_question_word(words) and len(words) >= 4:
            return True
        
        return False
    
    def _starts_with_question_word(self, words: List[str]) -> bool:
        """Check if starts with a question word."""
        return words and words[0] in self.question_words
    
    def _missing_legal_context(self, words: List[str]) -> bool:
        """Check if query lacks legal context."""
        # If it's a question word but no legal nouns
        if self._starts_with_question_word(words):
            has_legal = any(noun in words for noun in self.legal_nouns)
            if not has_legal and len(words) < 5:
                return True
        return False
    
    def _has_pronoun_without_reference(self, words: List[str]) -> bool:
        """Check if query has pronouns without clear reference."""
        pronouns = {"it", "its", "them", "they", "those", "these", "that"}
        return any(p in words for p in pronouns)


# ──────────────────────────────────────────────────────────────
# RESPONSE TEMPLATE GENERATOR (NEW MODULE)
# ──────────────────────────────────────────────────────────────

class ResponseTemplateGenerator:
    """Generates response templates based on intent."""
    
    def __init__(self):
        self.templates = {
            QueryIntent.DEFINITION: {
                "style": ResponseStyle.DEFINITION,
                "sections": [
                    "Definition",
                    "Purpose",
                    "Key Features",
                    "Example",
                    "Important Notes"
                ],
                "template": """
**Definition:**
{{definition}}

**Purpose:**
{{purpose}}

**Key Features:**
{{key_features}}

**Example:**
{{example}}

**Important Notes:**
{{important_notes}}
"""
            },
            QueryIntent.PROCEDURE: {
                "style": ResponseStyle.PROCEDURE,
                "sections": [
                    "Overview",
                    "Step 1",
                    "Step 2",
                    "Step 3",
                    "Documents Required",
                    "Time Limit",
                    "Fees"
                ],
                "template": """
**Overview:**
{{overview}}

**Steps:**
{{steps}}

**Documents Required:**
{{documents}}

**Time Limit:**
{{time_limit}}

**Fees:**
{{fees}}
"""
            },
            QueryIntent.RIGHTS: {
                "style": ResponseStyle.RIGHTS,
                "sections": [
                    "Rights",
                    "Conditions",
                    "Limitations",
                    "Authority",
                    "Appeal"
                ],
                "template": """
**Your Rights:**
{{rights}}

**Conditions:**
{{conditions}}

**Limitations:**
{{limitations}}

**Authority:**
{{authority}}

**Appeal:**
{{appeal}}
"""
            },
            QueryIntent.COMPLAINT: {
                "style": ResponseStyle.COMPLAINT,
                "sections": [
                    "Where to File",
                    "How to File",
                    "Documents Required",
                    "Timeline",
                    "Expected Outcome"
                ],
                "template": """
**Where to File:**
{{where}}

**How to File:**
{{how}}

**Documents Required:**
{{documents}}

**Timeline:**
{{timeline}}

**Expected Outcome:**
{{expected_outcome}}
"""
            },
            QueryIntent.COMPARISON: {
                "style": ResponseStyle.COMPARISON,
                "sections": [
                    "Key Differences",
                    "Similarities",
                    "When to Use",
                    "Examples"
                ],
                "template": """
**Key Differences:**
{{differences}}

**Similarities:**
{{similarities}}

**When to Use:**
{{when_to_use}}

**Examples:**
{{examples}}
"""
            },
            QueryIntent.PENALTY: {
                "style": ResponseStyle.PENALTY,
                "sections": [
                    "Penalty",
                    "Nature of Offense",
                    "Quantum",
                    "Exceptions",
                    "Appeal"
                ],
                "template": """
**Penalty:**
{{penalty}}

**Nature of Offense:**
{{nature}}

**Quantum:**
{{quantum}}

**Exceptions:**
{{exceptions}}

**Appeal:**
{{appeal}}
"""
            },
            QueryIntent.APPEAL: {
                "style": ResponseStyle.APPEAL,
                "sections": [
                    "Grounds",
                    "Procedure",
                    "Time Limit",
                    "Authority",
                    "Outcome"
                ],
                "template": """
**Grounds:**
{{grounds}}

**Procedure:**
{{procedure}}

**Time Limit:**
{{time_limit}}

**Authority:**
{{authority}}

**Outcome:**
{{outcome}}
"""
            },
            QueryIntent.CASE_ANALYSIS: {
                "style": ResponseStyle.CASE_ANALYSIS,
                "sections": [
                    "Facts",
                    "Issue",
                    "Arguments",
                    "Decision",
                    "Reasoning",
                    "Implications"
                ],
                "template": """
**Facts:**
{{facts}}

**Issue:**
{{issue}}

**Arguments:**
{{arguments}}

**Decision:**
{{decision}}

**Reasoning:**
{{reasoning}}

**Implications:**
{{implications}}
"""
            },
            QueryIntent.DECISION_SUPPORT: {
                "style": ResponseStyle.DECISION_SUPPORT,
                "sections": [
                    "Options",
                    "Risks",
                    "Benefits",
                    "Legal Considerations",
                    "Recommendation"
                ],
                "template": """
**Options:**
{{options}}

**Risks:**
{{risks}}

**Benefits:**
{{benefits}}

**Legal Considerations:**
{{legal_considerations}}

**Recommendation:**
{{recommendation}}
"""
            },
            QueryIntent.GENERAL: {
                "style": ResponseStyle.GENERAL,
                "sections": [
                    "Information",
                    "Sources",
                    "Disclaimer"
                ],
                "template": """
**Information:**
{{information}}

**Sources:**
{{sources}}

{{disclaimer}}
"""
            }
        }
    
    def get_template(self, intent: QueryIntent) -> Optional[Dict[str, Any]]:
        """Get template for intent."""
        return self.templates.get(intent, self.templates.get(QueryIntent.GENERAL))


# ──────────────────────────────────────────────────────────────
# KNOWLEDGE GAP DETECTOR (NEW MODULE)
# ──────────────────────────────────────────────────────────────

class KnowledgeGapDetector:
    """Detects knowledge gaps in the knowledge base."""
    
    def __init__(self):
        # Known legal topics that should be in KB
        self.known_topics = {
            "RTI": ["Right to Information Act", "RTI Act", "Information Commission"],
            "Contract": ["Indian Contract Act", "Agreement", "Breach of Contract"],
            "Cyber": ["Information Technology Act", "Cyber Fraud", "Cyber Crime"],
            "Employment": ["Industrial Disputes Act", "Employment Law", "Labor Law"],
            "Consumer": ["Consumer Protection Act", "Consumer Rights"],
            "Tenancy": ["Rent Control Act", "Tenancy Rights"],
            "Property": ["Transfer of Property Act", "Property Rights"],
            "Family": ["Hindu Marriage Act", "Divorce Law", "Custody Law"],
            "Criminal": ["Indian Penal Code", "CrPC", "Criminal Procedure"],
            "Corporate": ["Companies Act", "Corporate Law"],
        }
        
        # Legal concepts by domain
        self.domain_concepts = {
            "contract": ["offer", "acceptance", "consideration", "capacity", "void agreement", "breach", "remedy"],
            "employment": ["salary", "termination", "notice period", "workplace rights", "sexual harassment"],
            "rti": ["application", "appeal", "commission", "information", "public authority"],
            "cyber": ["digital signature", "electronic record", "intermediary", "data protection", "privacy"],
            "consumer": ["deficiency", "unfair trade practice", "warranty", "refund", "compensation"],
            "tenancy": ["rent", "deposit", "eviction", "notice", "lease agreement"],
            "property": ["ownership", "title", "possession", "encumbrance", "registration"],
            "family": ["maintenance", "custody", "alimony", "marriage", "divorce"],
            "criminal": ["arrest", "bail", "trial", "punishment", "appeal"],
            "corporate": ["director", "shareholder", "board meeting", "annual return", "audit"],
        }
    
    def detect_gaps(self, query: str, domain: Optional[str]) -> List[str]:
        """Detect knowledge gaps in the query."""
        if not domain:
            return []
        
        query_lower = query.lower()
        gaps = []
        
        # Check if query mentions specific topics not in KB
        if domain in self.domain_concepts:
            for concept in self.domain_concepts[domain]:
                if concept in query_lower:
                    # This concept might not be covered
                    gaps.append(f"Concept '{concept}' may need verification")
        
        # Check for known topics that might be missing
        if domain.upper() in self.known_topics:
            for topic in self.known_topics[domain.upper()]:
                if topic.lower() in query_lower:
                    # This specific law might not be in KB
                    gaps.append(f"Topic '{topic}' may require specific knowledge")
        
        return gaps
    
    def estimate_coverage(self, query: str, domain: Optional[str]) -> float:
        """Estimate KB coverage for the query."""
        if not domain:
            return 0.0
        
        query_lower = query.lower()
        total_concepts = 0
        matched_concepts = 0
        
        if domain in self.domain_concepts:
            concepts = self.domain_concepts[domain]
            total_concepts = len(concepts)
            
            for concept in concepts:
                if concept in query_lower:
                    matched_concepts += 1
        
        if total_concepts == 0:
            return 0.5
        
        return min(1.0, matched_concepts / total_concepts)


# ──────────────────────────────────────────────────────────────
# BASE MODULES (Enhanced)
# ──────────────────────────────────────────────────────────────

class InputValidator:
    """Handles basic input validation and sanitization."""
    
    def __init__(
        self,
        min_length: int = 3,
        max_length: int = 1000,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.incomplete_detector = IncompleteQueryDetector()
    
    def preprocess(self, text: str) -> str:
        """Preprocess input text."""
        if not text:
            return ""
        
        text = self._sanitize(text)
        text = self._remove_extra_spaces(text)
        text = self._remove_control_characters(text)
        
        if not self._validate_unicode(text):
            return ""
        
        return text
    
    def validate(self, text: str) -> Tuple[bool, List[str], float]:
        """Validate input text and return completeness score."""
        warnings = []
        
        if not text or not text.strip():
            warnings.append("empty_query")
            return False, warnings, 0.0
        
        if len(text.strip()) < self.min_length:
            warnings.append(f"too_short_min_{self.min_length}")
            return False, warnings, 0.0
        
        if len(text) > self.max_length:
            warnings.append(f"too_long_max_{self.max_length}")
            return False, warnings, 0.0
        
        if self._contains_only_symbols(text):
            warnings.append("only_symbols")
            return False, warnings, 0.0
        
        if self._contains_only_numbers(text):
            warnings.append("only_numbers")
            return False, warnings, 0.0
        
        if self._is_mixed_garbage(text):
            warnings.append("mixed_garbage")
            return False, warnings, 0.0
        
        # Check completeness
        is_incomplete, reason, completeness = self.incomplete_detector.detect(text)
        if is_incomplete:
            warnings.append(f"incomplete: {reason}")
            # Allow incomplete queries to pass but with warning
            return True, warnings, completeness
        
        return True, warnings, 1.0
    
    def repair_query(self, text: str) -> Optional[str]:
        """Attempt to repair common typos and issues."""
        if not text:
            return None
        
        # Common legal typos
        corrections = {
            "contrct": "contract",
            "conract": "contract",
            "contrac": "contract",
            "contarct": "contract",
            "salery": "salary",
            "sallery": "salary",
            "terminatione": "termination",
            "termnation": "termination",
            "wht": "what",
            "rti": "RTI",
            "r.t.i": "RTI",
            "empliyment": "employment",
            "employement": "employment",
            "tenent": "tenant",
            "landloard": "landlord",
            "evicition": "eviction",
            "compansation": "compensation",
            "deposite": "deposit",
            "pleaese": "please",
            "thnks": "thanks",
        }
        
        text_lower = text.lower()
        for wrong, correct in corrections.items():
            if wrong in text_lower:
                text = text.replace(wrong, correct)
        
        return text
    
    def _sanitize(self, text: str) -> str:
        """Remove dangerous characters."""
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
        return text.strip()
    
    def _remove_extra_spaces(self, text: str) -> str:
        """Remove extra spaces."""
        return re.sub(r"\s+", " ", text).strip()
    
    def _remove_control_characters(self, text: str) -> str:
        """Remove control characters."""
        return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    
    def _validate_unicode(self, text: str) -> bool:
        """Check Unicode validity."""
        try:
            text.encode("utf-8")
            return True
        except UnicodeEncodeError:
            return False
    
    def _contains_only_symbols(self, text: str) -> bool:
        """Check if text contains only symbols."""
        return bool(text) and all(not ch.isalnum() for ch in text if ch != " ")
    
    def _contains_only_numbers(self, text: str) -> bool:
        """Check if text contains only numbers."""
        text_clean = re.sub(r"\s+", "", text)
        return text_clean.isdigit() if text_clean else False
    
    def _is_mixed_garbage(self, text: str) -> bool:
        """Check if text is mixed garbage."""
        words = re.findall(r"\b\w+\b", text)
        if not words:
            return False
        
        # Check for keyboard smash
        if re.search(r"([a-z])\1{4,}", text.lower()):
            return True
        
        # Check for random characters
        avg_word_length = sum(len(w) for w in words) / len(words)
        return avg_word_length > 15


class SecurityValidator:
    """Handles security checks including prompt injection, SQL, XSS."""
    
    def __init__(self):
        self.prompt_injection_patterns = [
            r"ignore previous",
            r"forget your",
            r"you are now",
            r"system prompt",
            r"roleplay",
            r"jailbreak",
            r"you must",
            r"you are programmed",
            r"override",
            r"bypass",
            r"unrestricted",
            r"break out",
            r"escape",
        ]
        
        self.sql_patterns = [
            r"SELECT.*FROM",
            r"INSERT.*INTO",
            r"DROP.*TABLE",
            r"UNION.*SELECT",
            r"--",
            r";",
        ]
        
        self.xss_patterns = [
            r"<script",
            r"javascript:",
            r"onerror=",
            r"onload=",
            r"alert\(",
        ]
        
        self.command_patterns = [
            r"os\.system",
            r"subprocess",
            r"eval\(",
            r"exec\(",
            r"__import__",
        ]
    
    def validate(self, text: str) -> Tuple[bool, List[str]]:
        """Validate security of input."""
        warnings = []
        
        if self._detect_prompt_injection(text):
            warnings.append("prompt_injection")
            return False, warnings
        
        if self._detect_sql_injection(text):
            warnings.append("sql_injection")
            return False, warnings
        
        if self._detect_xss(text):
            warnings.append("xss")
            return False, warnings
        
        if self._detect_command_execution(text):
            warnings.append("command_execution")
            return False, warnings
        
        return True, warnings
    
    def _detect_prompt_injection(self, text: str) -> bool:
        """Detect prompt injection attempts."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.prompt_injection_patterns)
    
    def _detect_sql_injection(self, text: str) -> bool:
        """Detect SQL injection attempts."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.sql_patterns)
    
    def _detect_xss(self, text: str) -> bool:
        """Detect XSS attempts."""
        return any(re.search(pattern, text) for pattern in self.xss_patterns)
    
    def _detect_command_execution(self, text: str) -> bool:
        """Detect command execution attempts."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.command_patterns)


class IntentClassifier:
    """Classifies query intent using scoring-based approach with hierarchy."""
    
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.DEFINITION: {
                "keywords": ["what is", "define", "meaning of", "definition", "what are"],
                "weights": [40, 35, 30, 35, 30],
                "bonus": {"question": 5, "noun": 10}
            },
            QueryIntent.EXPLANATION: {
                "keywords": ["explain", "describe", "elaborate", "clarify", "tell me about"],
                "weights": [40, 35, 30, 25, 25],
                "bonus": {"question": 5, "complex_word": 10}
            },
            QueryIntent.PROCEDURE: {
                "keywords": ["how to", "how do i", "procedure", "steps", "process", "method"],
                "weights": [45, 40, 35, 30, 30, 25],
                "bonus": {"action": 15, "imperative": 10}
            },
            QueryIntent.RIGHTS: {
                "keywords": ["rights", "entitled", "eligible", "claim", "can i", "allowed"],
                "weights": [40, 35, 30, 30, 25, 20],
                "bonus": {"legal_entity": 15, "question": 5}
            },
            QueryIntent.COMPLAINT: {
                "keywords": ["complaint", "report", "file", "register", "lodge"],
                "weights": [40, 35, 30, 25, 25],
                "bonus": {"action": 15, "problem": 10}
            },
            QueryIntent.COMPARISON: {
                "keywords": ["difference", "compare", "vs", "versus", "between"],
                "weights": [40, 35, 30, 30, 25],
                "bonus": {"multiple_entities": 20}
            },
            QueryIntent.PENALTY: {
                "keywords": ["penalty", "fine", "punishment", "consequence"],
                "weights": [40, 35, 30, 25],
                "bonus": {"legal_entity": 10, "amount": 15}
            },
            QueryIntent.PUNISHMENT: {
                "keywords": ["punishment", "imprisonment", "jail", "prison", "sentence"],
                "weights": [40, 35, 30, 25, 20],
                "bonus": {"legal_entity": 10}
            },
            QueryIntent.APPEAL: {
                "keywords": ["appeal", "challenge", "object", "review", "appeal against"],
                "weights": [40, 35, 30, 25, 20],
                "bonus": {"legal_process": 15}
            },
            QueryIntent.DOCUMENT: {
                "keywords": ["document", "form", "application", "letter", "notice"],
                "weights": [40, 35, 30, 25, 20],
                "bonus": {"action": 10, "legal_entity": 10}
            },
            QueryIntent.EXAMPLES: {
                "keywords": ["example", "instance", "sample", "case", "scenario"],
                "weights": [40, 35, 30, 25, 20],
                "bonus": {"question": 5}
            },
            QueryIntent.CHECKLIST: {
                "keywords": ["checklist", "list", "items", "requirements", "needed"],
                "weights": [40, 30, 25, 20, 15],
                "bonus": {"action": 10}
            },
            QueryIntent.ELIGIBILITY: {
                "keywords": ["eligible", "qualify", "who can", "criteria", "requirements"],
                "weights": [40, 35, 30, 25, 20],
                "bonus": {"question": 5, "legal_entity": 10}
            },
            QueryIntent.TIME_LIMIT: {
                "keywords": ["time limit", "deadline", "period", "duration", "within", "timeline"],
                "weights": [40, 35, 30, 25, 20, 15],
                "bonus": {"number": 10, "time": 15}
            },
            QueryIntent.AUTHORITY: {
                "keywords": ["authority", "officer", "official", "department", "agency"],
                "weights": [40, 35, 30, 25, 20],
                "bonus": {"legal_entity": 10}
            },
            QueryIntent.COURT: {
                "keywords": ["court", "judge", "tribunal", "bench", "hearing", "magistrate"],
                "weights": [40, 35, 30, 25, 20, 15],
                "bonus": {"legal_entity": 10}
            },
            QueryIntent.SECTION: {
                "keywords": ["section", "clause", "article", "provision", "sub-section"],
                "weights": [40, 35, 30, 25, 20],
                "bonus": {"number": 15, "law": 10}
            },
            QueryIntent.FINE: {
                "keywords": ["fine", "amount", "compensation", "damages", "penalty"],
                "weights": [40, 35, 30, 25, 20],
                "bonus": {"amount": 15, "money": 10}
            },
            QueryIntent.FORMS: {
                "keywords": ["form", "application", "register", "submit", "fill"],
                "weights": [40, 35, 30, 25, 20],
                "bonus": {"action": 10}
            },
            QueryIntent.DRAFTING: {
                "keywords": ["draft", "prepare", "write", "create", "make", "compose"],
                "weights": [40, 35, 30, 25, 20, 15],
                "bonus": {"action": 15, "legal_document": 20}
            },
            QueryIntent.CASE_ANALYSIS: {
                "keywords": ["case", "judgment", "precedent", "analysis", "apply"],
                "weights": [40, 35, 30, 25, 20],
                "bonus": {"legal_entity": 10, "complex": 15}
            },
            QueryIntent.DECISION_SUPPORT: {
                "keywords": ["decide", "choose", "option", "should i", "whether"],
                "weights": [40, 35, 30, 25, 20],
                "bonus": {"question": 10, "complex": 15}
            },
            QueryIntent.SUMMARIZATION: {
                "keywords": ["summarize", "summary", "brief", "overview", "gist"],
                "weights": [40, 35, 30, 25, 20],
                "bonus": {"action": 10}
            },
            QueryIntent.RIGHTS_IDENTIFICATION: {
                "keywords": ["what rights", "my rights", "entitlements", "can i do"],
                "weights": [45, 40, 35, 30],
                "bonus": {"legal_entity": 10, "question": 5}
            },
            QueryIntent.RISK_ANALYSIS: {
                "keywords": ["risk", "exposure", "liability", "danger", "threat"],
                "weights": [40, 35, 30, 25, 20],
                "bonus": {"legal_entity": 10, "complex": 15}
            },
        }
        
        # Common stopwords for scoring
        self.stopwords = {"a", "an", "the", "to", "for", "of", "with", "on", "at", "by", "from"}
        self.legal_entities = {"act", "section", "court", "authority", "tribunal", "commission"}
    
    def classify(self, text: str) -> Tuple[QueryIntent, Dict[str, float], List[Tuple[str, float]]]:
        """Classify intent with hierarchy."""
        text_lower = text.lower()
        tokens = set(re.findall(r"\b\w+\b", text_lower))
        
        scores = {}
        
        for intent, config in self.intent_patterns.items():
            score = 0
            
            # Check keywords
            for idx, keyword in enumerate(config["keywords"]):
                if keyword in text_lower:
                    score += config["weights"][idx] if idx < len(config["weights"]) else 20
            
            # Apply bonuses (same as before)
            score += self._apply_bonuses(text_lower, tokens, config.get("bonus", {}))
            
            scores[intent.value] = min(score, 100)
        
        # Sort intents by score
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_intents and sorted_intents[0][1] > 30:
            # Find the matching QueryIntent
            for intent in QueryIntent:
                if intent.value == sorted_intents[0][0]:
                    # Get hierarchy (top 3)
                    hierarchy = sorted_intents[:3]
                    return intent, scores, hierarchy
        
        return QueryIntent.UNKNOWN, scores, []
    
    def _apply_bonuses(self, text: str, tokens: Set[str], bonuses: Dict[str, int]) -> int:
        """Apply bonus scores."""
        score = 0
        
        if "question" in bonuses and text.endswith("?"):
            score += bonuses["question"]
        
        if "noun" in bonuses:
            nouns = {"right", "law", "court", "agreement", "contract", "procedure"}
            if any(noun in tokens for noun in nouns):
                score += bonuses["noun"]
        
        if "legal_entity" in bonuses:
            if any(entity in tokens for entity in self.legal_entities):
                score += bonuses["legal_entity"]
        
        if "action" in bonuses:
            action_words = {"file", "register", "submit", "apply", "appeal", "complain"}
            if any(action in tokens for action in action_words):
                score += bonuses["action"]
        
        if "amount" in bonuses:
            if re.search(r"\b\d+\s*(lakh|crore|thousand|hundred|rupees?)\b", text):
                score += bonuses["amount"]
        
        if "number" in bonuses:
            if re.search(r"\b\d+\b", text):
                score += bonuses["number"]
        
        if "multiple_entities" in bonuses:
            entities = re.findall(r"\b(act|court|section|rule|regulation)s?\b", text)
            if len(set(entities)) >= 2:
                score += bonuses["multiple_entities"]
        
        if "complex" in bonuses:
            word_count = len(text.split())
            if word_count > 20:
                score += bonuses["complex"]
        
        if "law" in bonuses:
            if any(word in tokens for word in ["act", "law", "constitution", "regulation"]):
                score += bonuses["law"]
        
        if "time" in bonuses:
            time_words = {"day", "month", "year", "week", "hour", "minute"}
            if any(word in tokens for word in time_words):
                score += bonuses["time"]
        
        if "money" in bonuses:
            money_words = {"rupee", "dollar", "pound", "euro", "currency"}
            if any(word in tokens for word in money_words):
                score += bonuses["money"]
        
        if "imperative" in bonuses:
            if text.startswith(("tell", "show", "give", "provide")):
                score += bonuses["imperative"]
        
        if "legal_document" in bonuses:
            legal_docs = {"contract", "agreement", "notice", "deed", "will", "trust"}
            if any(doc in tokens for doc in legal_docs):
                score += bonuses["legal_document"]
        
        if "complex_word" in bonuses:
            complex_words = {"notwithstanding", "herein", "thereof", "whereas"}
            if any(word in tokens for word in complex_words):
                score += bonuses["complex_word"]
        
        if "legal_process" in bonuses:
            process_words = {"appeal", "review", "hearing", "trial", "proceeding"}
            if any(word in tokens for word in process_words):
                score += bonuses["legal_process"]
        
        if "problem" in bonuses:
            problem_words = {"issue", "problem", "concern", "complaint", "grievance"}
            if any(word in tokens for word in problem_words):
                score += bonuses["problem"]
        
        return score


class DomainClassifier:
    """Classifies legal domain with hierarchy."""
    
    def __init__(self, supported_domains: Optional[List[str]] = None):
        self.supported_domains = supported_domains or [
            "contract", "employment", "rti", "cyber", "consumer",
            "tenancy", "tenant", "property", "women", "senior", "motor_vehicle"
        ]
        
        self.domain_keywords = {
            "contract": {
                "keywords": ["contract", "agreement", "terms", "conditions", "breach", "clause"],
                "weights": [40, 35, 30, 25, 35, 30]
            },
            "employment": {
                "keywords": ["employment", "salary", "termination", "worker", "employee", "job"],
                "weights": [40, 35, 35, 30, 30, 25]
            },
            "rti": {
                "keywords": ["rti", "right to information", "information", "transparency", "disclosure"],
                "weights": [45, 40, 30, 25, 20]
            },
            "cyber": {
                "keywords": [
                    "cyber", "online", "digital", "fraud", "upi", "phishing", "hacking",
                    "phonepe", "google pay", "gpay", "paytm", "otp", "upi pin",
                    "scam", "tricked", "unauthorized transaction"
                ],
                "weights": [40, 35, 30, 35, 35, 25, 20, 35, 35, 35, 30, 30, 30, 35, 35]
            },
            "consumer": {
                "keywords": ["consumer", "product", "refund", "defective", "warranty", "service"],
                "weights": [40, 35, 30, 25, 25, 20]
            },
            "tenancy": {
                "keywords": ["rent", "tenant", "landlord", "eviction", "deposit", "lease"],
                "weights": [40, 35, 35, 30, 25, 25]
            },
            "property": {
                "keywords": ["property", "ownership", "title", "deed", "registration", "land"],
                "weights": [40, 35, 30, 25, 20, 15]
            },
            "women": {
                "keywords": ["women", "woman", "harassment", "domestic violence", "stalking", "workplace harassment"],
                "weights": [40, 40, 35, 35, 25, 30]
            },
            "senior": {
                "keywords": ["senior citizen", "elder", "elderly", "maintenance of parents", "parent maintenance"],
                "weights": [40, 30, 30, 35, 35]
            },
            "motor_vehicle": {
                "keywords": ["motor vehicle", "traffic", "challan", "driving licence", "driving license", "accident"],
                "weights": [40, 25, 30, 30, 30, 25]
            },
            "family": {
                "keywords": ["marriage", "divorce", "custody", "maintenance", "child", "spouse"],
                "weights": [40, 35, 30, 25, 20, 15]
            },
            "criminal": {
                "keywords": ["crime", "police", "fir", "arrest", "bail", "offence"],
                "weights": [40, 35, 30, 25, 20, 15]
            },
            "corporate": {
                "keywords": ["company", "shareholder", "director", "incorporation", "board"],
                "weights": [40, 35, 30, 25, 20]
            }
        }
    
    def classify(self, text: str) -> Tuple[Optional[str], Dict[str, float], List[Tuple[str, float]]]:
        """Classify domain with hierarchy."""
        text_lower = text.lower()
        
        scores = {}
        
        for domain, config in self.domain_keywords.items():
            score = 0
            
            for idx, keyword in enumerate(config["keywords"]):
                if keyword in text_lower:
                    score += config["weights"][idx] if idx < len(config["weights"]) else 20
            
            scores[domain] = min(score, 100)
        
        # Sort domains by score
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter by threshold
        filtered = [(d, s) for d, s in sorted_domains if s > 20]
        
        if filtered:
            # Return top domain, all scores, and hierarchy
            return filtered[0][0], scores, filtered[:3]
        
        return None, scores, []
    
    def is_supported(self, domain: str) -> bool:
        """Check if domain is supported."""
        return domain in self.supported_domains if domain else False
    
    def get_supported_topics(self) -> List[str]:
        """Get supported topics."""
        return self.supported_domains


class EntityExtractor:
    """Extracts legal entities with semantic concepts."""
    
    def __init__(self):
        self.act_patterns = [
            r"(\w+\s+Act)",
            r"(\w+-\w+)\s+Act",
            r"(RTI|IPC|CrPC|IT)\s+Act",
        ]
        
        self.section_patterns = [
            r"Section\s+(\d+)",
            r"Sec\.\s+(\d+)",
            r"S\.\s+(\d+)",
            r"Article\s+(\d+)",
        ]
        
        self.court_patterns = [
            r"(Supreme Court|High Court|District Court|Session Court)",
            r"Court of\s+(\w+)",
            r"(\w+)\s+Tribunal",
        ]
        
        self.authority_patterns = [
            r"(Commission|Board|Authority|Department|Ministry)",
            r"(\w+)\s+Commission",
            r"(\w+)\s+Authority",
        ]
        
        self.time_patterns = [
            r"(\d+)\s*(days?|months?|years?|weeks?|hours?)",
            r"within\s+(\d+)\s*(days?|months?)",
            r"period of\s+(\d+)\s*(days?|months?)",
        ]
        
        self.penalty_patterns = [
            r"imprisonment for\s+(\d+)\s*(years?|months?)",
            r"fine of\s+([\d,]+)\s*(rupees?|lakh|crore)",
            r"penalty of\s+([\d,]+)",
        ]
        
        self.rights_patterns = [
            r"right to\s+(\w+)",
            r"rights? to\s+(\w+)",
            r"entitled to\s+(\w+)",
        ]
        
        self.form_patterns = [
            r"Form\s+(\w+)",
            r"Application\s+for\s+(\w+)",
            r"(\w+)\s+Application",
        ]
        
        # Semantic legal concepts
        self.concept_patterns = {
            "offer": r"\boffer\b",
            "acceptance": r"\bacceptance\b",
            "consideration": r"\bconsideration\b",
            "capacity": r"\bcapacity\b",
            "void agreement": r"\bvoid\s+agreement\b",
            "breach": r"\bbreach\b",
            "remedy": r"\bremedy\b",
            "termination": r"\btermination\b",
            "compensation": r"\bcompensation\b",
            "damages": r"\bdamages\b",
            "injunction": r"\binjunction\b",
            "specific performance": r"\bspecific\s+performance\b",
        }
    
    def extract(self, text: str) -> ExtractedEntities:
        """Extract all entities including semantic concepts."""
        entities = ExtractedEntities()
        
        # Extract acts
        for pattern in self.act_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.acts.extend(matches)
        
        # Extract sections
        for pattern in self.section_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.sections.extend(matches)
        
        # Extract courts
        for pattern in self.court_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.courts.extend(matches)
        
        # Extract authorities
        for pattern in self.authority_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.authorities.extend(matches)
        
        # Extract time periods
        for pattern in self.time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.time_periods.extend(matches)
        
        # Extract penalties
        for pattern in self.penalty_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.penalties.extend(matches)
        
        # Extract rights
        for pattern in self.rights_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.rights.extend(matches)
        
        # Extract forms
        for pattern in self.form_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.forms.extend(matches)
        
        # Extract amounts
        amount_pattern = r"[\d,]+\s*(rupees?|rs\.?|₹|lakh|crore)"
        entities.amounts = re.findall(amount_pattern, text, re.IGNORECASE)
        
        # Extract dates
        date_pattern = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}"
        entities.dates = re.findall(date_pattern, text, re.IGNORECASE)
        
        # Extract legal concepts
        for concept, pattern in self.concept_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                entities.legal_concepts.append(concept)
        
        return entities


class ConversationTracker:
    """Tracks conversation state with follow-up resolution."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = SessionState(session_id=session_id)
        
        # Pronoun references
        self.pronoun_references = {
            "it": None,
            "its": None,
            "them": None,
            "they": None,
            "those": None,
            "these": None,
            "that": None,
        }
    
    def update(self, result: ValidationResult) -> None:
        """Update conversation state."""
        self.state.last_question = result.normalized_query
        self.state.last_domain = result.detected_domain
        self.state.last_domain_scores = result.domain_scores
        self.state.last_intent = result.detected_intent
        self.state.question_count += 1
        self.state.last_active = time.time()
        
        if result.is_followup:
            self.state.followup_count += 1
        
        if result.detected_domain:
            self.state.topic_history.append(result.detected_domain)
            if len(self.state.topic_history) > 10:
                self.state.topic_history = self.state.topic_history[-10:]
        
        if result.entities:
            self.state.entity_history.append(result.entities)
            if len(self.state.entity_history) > 5:
                self.state.entity_history = self.state.entity_history[-5:]
        
        # Update context for follow-up resolution
        if result.detected_domain:
            self.state.last_domain_context = result.detected_domain
        
        if result.entities.acts:
            self.state.last_entity_context = ", ".join(result.entities.acts[:3])
        elif result.entities.sections:
            self.state.last_entity_context = f"Section {result.entities.sections[0]}"
        
        if result.entities.legal_concepts:
            self.state.last_topic_context = ", ".join(result.entities.legal_concepts[:3])
    
    def is_followup(self, text: str) -> bool:
        """Detect if current text is a followup."""
        if not self.state.topic_history:
            return False
        
        # Check for references to previous topics
        text_lower = text.lower()
        for topic in self.state.topic_history[-3:]:
            if topic.lower() in text_lower:
                return True
        
        # Check for pronouns
        pronouns = {"it", "its", "them", "they", "that", "those", "these"}
        words = set(re.findall(r"\b\w+\b", text_lower))
        if pronouns.intersection(words):
            return True
        
        return False
    
    def resolve_followup(self, text: str) -> str:
        """Resolve followup query by adding context."""
        if not self.is_followup(text):
            return text
        
        text_lower = text.lower()
        resolved = text
        
        # Add domain context
        if self.state.last_domain_context and self.state.last_domain_context not in text_lower:
            resolved = f"{resolved} about {self.state.last_domain_context}"
        
        # Add entity context
        if self.state.last_entity_context and self.state.last_entity_context not in text_lower:
            resolved = f"{resolved} regarding {self.state.last_entity_context}"
        
        # Add topic context for pronouns
        if "it" in text_lower or "its" in text_lower:
            if self.state.last_topic_context:
                resolved = resolved.replace("it", f"this {self.state.last_topic_context}")
        
        return resolved
    
    def get_context(self) -> Dict[str, Any]:
        """Get conversation context."""
        return {
            "last_domain": self.state.last_domain,
            "last_intent": self.state.last_intent,
            "last_question": self.state.last_question,
            "topic_history": self.state.topic_history[-5:],
            "followup_count": self.state.followup_count,
            "question_count": self.state.question_count,
            "last_topic_context": self.state.last_topic_context,
            "last_entity_context": self.state.last_entity_context,
        }


class KnowledgeAwareness:
    """Handles knowledge base awareness with gap detection."""
    
    def __init__(self, supported_domains: List[str]):
        self.supported_domains = set(supported_domains)
        self.gap_detector = KnowledgeGapDetector()
        
        # Domain coverage estimation (0-1 scale)
        self.domain_coverage = {
            domain: 0.9 for domain in supported_domains  # High coverage for supported domains
        }
    
    def estimate_coverage(self, domain_scores: Dict[str, float]) -> float:
        """Estimate knowledge base coverage."""
        if not domain_scores:
            return 0.0
        
        relevant_domains = {d: s for d, s in domain_scores.items() if s > 10}
        if not relevant_domains:
            return 0.0
        
        total_score = sum(relevant_domains.values())
        weighted_coverage = 0.0
        
        for domain, score in relevant_domains.items():
            coverage = self.domain_coverage.get(domain, 0.0)
            weighted_coverage += (score / total_score) * coverage
        
        return min(weighted_coverage, 1.0)
    
    def detect_gaps(self, query: str, domain: Optional[str]) -> List[str]:
        """Detect knowledge gaps."""
        return self.gap_detector.detect_gaps(query, domain)
    
    def is_supported(self, domain: str) -> bool:
        """Check if domain is supported."""
        return domain in self.supported_domains
    
    def get_supported_topics(self) -> List[str]:
        """Get supported topics."""
        return sorted(self.supported_domains)


class QueryComplexityEstimator:
    """Estimates query complexity with reasoning depth."""
    
    def estimate(self, text: str, intent: QueryIntent, entities: ExtractedEntities) -> Tuple[QueryComplexity, int]:
        """Estimate complexity and reasoning depth."""
        word_count = len(text.split())
        token_count = len(re.findall(r"\b\w+\b", text))
        
        # Count legal entities
        entity_count = (
            len(entities.acts) + len(entities.sections) + len(entities.courts) +
            len(entities.authorities) + len(entities.procedures) + len(entities.legal_concepts)
        )
        
        # Check for legal jargon
        jargon_count = sum(1 for word in re.findall(r"\b\w+\b", text.lower())
                          if word in {"notwithstanding", "herein", "thereof", "whereas"})
        
        # Score
        score = 0
        reasoning_depth = 1  # Default
        
        if word_count > 20:
            score += 1
        if token_count > 30:
            score += 1
        if entity_count > 3:
            score += 1
        if jargon_count > 2:
            score += 1
        if intent in {QueryIntent.CASE_ANALYSIS, QueryIntent.DECISION_SUPPORT, QueryIntent.RISK_ANALYSIS}:
            score += 1
            reasoning_depth = 3  # Deep reasoning
        elif intent in {QueryIntent.COMPARISON, QueryIntent.COMPLAINT, QueryIntent.APPEAL}:
            reasoning_depth = 2  # Medium reasoning
        
        if score <= 1:
            return QueryComplexity.SIMPLE, reasoning_depth
        elif score <= 3:
            return QueryComplexity.MEDIUM, reasoning_depth
        elif score <= 4:
            return QueryComplexity.COMPLEX, reasoning_depth
        else:
            return QueryComplexity.EXPERT, reasoning_depth + 1


class ResponseTemplateGenerator:
    """Generates response templates and styles."""

    # NOTE: templates are defined in the earlier ResponseTemplateGenerator class body.
    # This second class overrides the name, so avoid any self-references.
    def __init__(self):
        # Keep a local reference to templates defined earlier in this file.
        # The class body above already defines `self.templates`; this __init__ should not call itself.
        self.templates = self.__class__.templates if hasattr(self.__class__, 'templates') else {}


    def get_template(self, intent: QueryIntent) -> Optional[Dict[str, Any]]:
        """Get template for intent."""
        return self.templates.get(intent, self.templates.get(QueryIntent.GENERAL))

    
    def get_response_style(self, intent: QueryIntent) -> ResponseStyle:
        """Get response style for intent."""
        template = self.get_template(intent)
        return template["style"] if template else ResponseStyle.GENERAL


class QueryRewriter:
    """Rewrites queries for better retrieval."""
    
    def __init__(self):
        self.pronoun_map = {
            "it": None,
            "its": None,
            "them": None,
            "they": None,
            "those": None,
            "these": None,
            "that": None,
        }
    
    def rewrite(self, query: str, session: Optional[ConversationTracker] = None) -> str:
        """Rewrite query for retrieval."""
        if not session:
            return query
        
        # Resolve followup
        if session.is_followup(query):
            return session.resolve_followup(query)
        
        return query
    
    def extract_keywords(self, query: str, entities: ExtractedEntities) -> List[str]:
        """Extract keywords for retrieval."""
        keywords = []
        
        # Add entities
        keywords.extend(entities.acts)
        keywords.extend(entities.sections)
        keywords.extend(entities.legal_concepts)
        keywords.extend(entities.rights)
        
        # Add key terms from query
        words = re.findall(r"\b\w+\b", query.lower())
        stopwords = {"what", "is", "are", "the", "a", "an", "to", "for", "of", "with", "on", "at"}
        
        for word in words:
            if len(word) > 3 and word not in stopwords:
                keywords.append(word)
        
        # Remove duplicates and limit
        return list(dict.fromkeys(keywords))[:10]


# ──────────────────────────────────────────────────────────────
# MAIN VALIDATOR (Enhanced)
# ──────────────────────────────────────────────────────────────

class QueryValidator:
    """Main query validator with all enhanced features."""
    
    def __init__(
        self,
        min_length: int = 3,
        max_length: int = 1000,
        supported_domains: Optional[List[str]] = None,
        max_unrelated_warnings: int = 3,
        max_clarifications: int = 3,
        enable_repair: bool = True,
    ):
        self.max_unrelated_warnings = max_unrelated_warnings
        self.max_clarifications = max_clarifications
        self.enable_repair = enable_repair
        
        # Initialize modules
        self.input_validator = InputValidator(min_length, max_length)
        self.security_validator = SecurityValidator()
        self.intent_classifier = IntentClassifier()
        self.domain_classifier = DomainClassifier(supported_domains)
        self.entity_extractor = EntityExtractor()
        self.knowledge_awareness = KnowledgeAwareness(
            supported_domains or self.domain_classifier.supported_domains
        )
        self.complexity_estimator = QueryComplexityEstimator()
        self.template_generator = ResponseTemplateGenerator()
        self.query_rewriter = QueryRewriter()
        
        # Sessions
        self.sessions: Dict[str, ConversationTracker] = {}
    
    def validate(self, query: str, session_id: Optional[str] = None) -> ValidationResult:
        """Main validation pipeline."""
        start_time = time.time()
        logs = []
        
        # ── 1. Preprocess ──
        preprocessed = self.input_validator.preprocess(query)
        logs.append({"step": "preprocess", "value": preprocessed})
        
        # ── 2. Repair if enabled ──
        corrected = None
        if self.enable_repair:
            corrected = self.input_validator.repair_query(preprocessed)
            if corrected and corrected != preprocessed:
                logs.append({"step": "repair", "original": preprocessed, "corrected": corrected})
                preprocessed = corrected
        
        # ── 3. Validate ──
        valid, warnings, completeness = self.input_validator.validate(preprocessed)
        logs.append({"step": "validate", "valid": valid, "warnings": warnings, "completeness": completeness})
        
        if not valid:
            return ValidationResult(
                valid=False,
                action=ValidationAction.REJECT,
                normalized_query=preprocessed,
                corrected_query=corrected,
                reason=warnings[0] if warnings else "invalid",
                response="Invalid query. Please ask a proper legal question.",
                warnings=warnings,
                processing_time_ms=(time.time() - start_time) * 1000,
                logs=logs,
            )
        
        # ── 4. Security check ──
        secure, security_warnings = self.security_validator.validate(preprocessed)
        logs.append({"step": "security", "secure": secure, "warnings": security_warnings})
        
        if not secure:
            return ValidationResult(
                valid=False,
                action=ValidationAction.REJECT,
                normalized_query=preprocessed,
                corrected_query=corrected,
                reason=security_warnings[0] if security_warnings else "security",
                response="Security check failed. Please ask a legitimate legal question.",
                warnings=security_warnings,
                processing_time_ms=(time.time() - start_time) * 1000,
                logs=logs,
            )
        
        # ── 5. Get or create session ──
        session = None
        if session_id:
            if session_id not in self.sessions:
                self.sessions[session_id] = ConversationTracker(session_id)
            session = self.sessions[session_id]
        
        # ── 6. Detect domains with hierarchy ──
        domain, domain_scores, domain_hierarchy = self.domain_classifier.classify(preprocessed)
        logs.append({"step": "domain", "domain": domain, "scores": domain_scores, "hierarchy": domain_hierarchy})
        
        # ── 7. Detect intent with hierarchy ──
        intent, intent_scores, intent_hierarchy = self.intent_classifier.classify(preprocessed)
        logs.append({"step": "intent", "intent": intent.value, "scores": intent_scores, "hierarchy": intent_hierarchy})
        
        # Set primary, secondary, tertiary intents
        primary_intent = intent
        secondary_intent = None
        tertiary_intent = None
        
        if len(intent_hierarchy) >= 2:
            for q_intent in QueryIntent:
                if q_intent.value == intent_hierarchy[1][0]:
                    secondary_intent = q_intent
                    break
        
        if len(intent_hierarchy) >= 3:
            for q_intent in QueryIntent:
                if q_intent.value == intent_hierarchy[2][0]:
                    tertiary_intent = q_intent
                    break
        
        # ── 8. Extract entities ──
        entities = self.entity_extractor.extract(preprocessed)
        logs.append({"step": "entities", "entities": entities})
        
        # ── 9. Conversation tracking ──
        is_followup = False
        query_rewrite = preprocessed
        if session:
            is_followup = session.is_followup(preprocessed)
            # Rewrite query for retrieval
            query_rewrite = self.query_rewriter.rewrite(preprocessed, session)
            logs.append({"step": "conversation", "is_followup": is_followup, "context": session.get_context()})
        
        # ── 10. Extract retrieval keywords ──
        retrieval_keywords = self.query_rewriter.extract_keywords(query_rewrite, entities)
        logs.append({"step": "keywords", "keywords": retrieval_keywords})
        
        # ── 11. Detect knowledge gaps ──
        missing_knowledge = self.knowledge_awareness.detect_gaps(preprocessed, domain)
        logs.append({"step": "knowledge_gaps", "gaps": missing_knowledge})
        
        # ── 12. Determine action ──
        action, reason, action_response = self._determine_action(
            preprocessed, domain, domain_scores, intent, session, missing_knowledge
        )
        logs.append({"step": "action", "action": action.value, "reason": reason})
        
        # ── 13. Calculate confidence ──
        confidence = self._calculate_confidence(
            preprocessed, domain_scores, intent_scores, domain, intent, session, completeness
        )
        logs.append({"step": "confidence", "confidence": confidence})
        
        # ── 14. Check KB coverage ──
        kb_coverage = self.knowledge_awareness.estimate_coverage(domain_scores)
        kb_has_knowledge = kb_coverage > 0.3 and len(missing_knowledge) == 0
        logs.append({"step": "kb_coverage", "coverage": kb_coverage, "has_knowledge": kb_has_knowledge})
        
        # ── 15. Determine retrieval strategy ──
        retrieval_strategy = self._determine_retrieval_strategy(
            action, domain, kb_coverage, intent
        )
        logs.append({"step": "retrieval", "strategy": retrieval_strategy.value})
        
        # ── 16. Estimate query complexity ──
        complexity, reasoning_depth = self.complexity_estimator.estimate(preprocessed, intent, entities)
        logs.append({"step": "complexity", "complexity": complexity.value, "reasoning_depth": reasoning_depth})
        
        # ── 17. Get response template ──
        response_style = self.template_generator.get_response_style(intent)
        template = self.template_generator.get_template(intent)
        logs.append({"step": "response_style", "style": response_style.value})
        
        # ── 18. Build response ──
        response = ""
        if action != ValidationAction.CONTINUE:
            response = self._build_response(action, reason, domain)
        
        # ── 19. Update session ──
        if session:
            # Create temporary result for session update
            temp_result = ValidationResult(
                valid=True,
                action=action,
                normalized_query=preprocessed,
                detected_domain=domain,
                domain_scores=domain_scores,
                detected_intent=intent,
                is_followup=is_followup,
                entities=entities,
            )
            session.update(temp_result)
        
        # ── 20. Build final result ──
        return ValidationResult(
            valid=True,
            action=action,
            normalized_query=preprocessed,
            corrected_query=corrected,
            query_rewrite=query_rewrite if query_rewrite != preprocessed else None,
            retrieval_keywords=retrieval_keywords,
            reason=reason,
            response=response,
            confidence=confidence,
            detected_domain=domain,
            domain_scores=domain_scores,
            domain_hierarchy=domain_hierarchy,
            detected_intent=intent,
            intent_scores=intent_scores,
            intent_hierarchy=intent_hierarchy,
            primary_intent=primary_intent,
            secondary_intent=secondary_intent,
            tertiary_intent=tertiary_intent,
            response_style=response_style,
            response_template=template["template"] if template else None,
            emotional_state=EmotionalState.NEUTRAL,
            is_followup=is_followup,
            retrieval_strategy=retrieval_strategy,
            kb_has_knowledge=kb_has_knowledge,
            entities=entities,
            query_complexity=complexity,
            missing_knowledge=missing_knowledge,
            suggested_topics=self.knowledge_awareness.get_supported_topics(),
            warnings=warnings,
            processing_time_ms=(time.time() - start_time) * 1000,
            logs=logs,
        )
    
    def _determine_action(
        self,
        text: str,
        domain: Optional[str],
        domain_scores: Dict[str, float],
        intent: QueryIntent,
        session: Optional[ConversationTracker],
        missing_knowledge: List[str],
    ) -> Tuple[ValidationAction, str, str]:
        """Determine the appropriate action based on classification."""
        text_lower = text.lower()
        
        # Greeting detection
        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
        if text_lower.strip() in greetings or text_lower.startswith(tuple(greetings)):
            return ValidationAction.GREETING, "greeting", ""
        
        # Goodbye detection
        goodbyes = {"bye", "goodbye", "see you", "thanks", "thank you"}
        if any(g in text_lower for g in goodbyes) and len(text_lower.split()) < 5:
            return ValidationAction.GOODBYE, "goodbye", ""
        
        # Non-legal domain
        non_legal = self._detect_non_legal(text_lower)
        if non_legal:
            if session:
                session.state.unrelated_questions += 1
                if session.state.unrelated_questions >= self.max_unrelated_warnings:
                    return ValidationAction.REJECT, "exceeded_warnings", ""
            return ValidationAction.CLARIFY, "non_legal_domain", ""
        
        # Knowledge gaps - critical!
        if missing_knowledge and len(missing_knowledge) > 0:
            return ValidationAction.CLARIFY, "knowledge_gap", ""
        
        # Unsupported domain
        if domain and not self.knowledge_awareness.is_supported(domain):
            if session:
                session.state.unrelated_questions += 1
                if session.state.unrelated_questions >= self.max_unrelated_warnings:
                    return ValidationAction.REJECT, "exceeded_warnings", ""
            return ValidationAction.CLARIFY, "unsupported_domain", ""
        
        # Low domain confidence
        max_score = max(domain_scores.values()) if domain_scores else 0
        if max_score < 20:
            return ValidationAction.CLARIFY, "low_confidence", ""
        
        # Incomplete query
        is_incomplete, _, _ = self.input_validator.incomplete_detector.detect(text)
        if is_incomplete:
            return ValidationAction.CLARIFY, "incomplete_query", ""
        
        # All checks passed
        return ValidationAction.CONTINUE, "valid_query", ""
    
    def _detect_non_legal(self, text: str) -> Optional[str]:
        """Detect non-legal domains."""
        programming = {"python", "javascript", "react", "docker", "api", "code"}
        engineering = {"engineer", "design", "machine", "circuit"}
        medical = {"doctor", "hospital", "disease", "medicine"}
        finance = {"stock", "market", "investment", "bond"}
        everyday = {"pizza", "recipe", "cook", "cooking", "cricket", "football", "movie", "song", "weather"}
        
        if any(kw in text for kw in programming):
            return "programming"
        if any(kw in text for kw in engineering):
            return "engineering"
        if any(kw in text for kw in medical):
            return "medical"
        if any(kw in text for kw in finance):
            return "finance"
        if any(kw in text for kw in everyday):
            return "non_legal"
        
        return None
    
    def _calculate_confidence(
        self,
        text: str,
        domain_scores: Dict[str, float],
        intent_scores: Dict[str, float],
        domain: Optional[str],
        intent: QueryIntent,
        session: Optional[ConversationTracker],
        completeness: float,
    ) -> ConfidenceScores:
        """Calculate comprehensive confidence scores."""
        confidence = ConfidenceScores()
        
        # Input quality
        word_count = len(text.split())
        if word_count >= 3:
            confidence.input_quality = min(1.0, word_count / 10)
        else:
            confidence.input_quality = 0.3
        
        # Completeness score
        confidence.completeness_score = completeness
        
        # Domain confidence
        if domain and domain_scores:
            confidence.domain_confidence = min(1.0, domain_scores.get(domain, 0) / 100)
        else:
            confidence.domain_confidence = 0.0
        
        # Intent confidence
        if intent and intent_scores:
            confidence.intent_confidence = min(1.0, intent_scores.get(intent.value, 0) / 100)
        else:
            confidence.intent_confidence = 0.0
        
        # Knowledge coverage
        confidence.knowledge_coverage = self.knowledge_awareness.estimate_coverage(domain_scores)
        
        # Conversation context
        if session and session.state.question_count > 1:
            confidence.conversation_context = 0.9
        else:
            confidence.conversation_context = 0.5
        
        # Metadata score (placeholder - would come from retriever)
        confidence.metadata_score = 0.5
        confidence.retrieval_score = 0.5
        confidence.embedding_similarity = 0.5
        
        # Calculate final confidence with completeness weight
        confidence.final_confidence = (
            confidence.input_quality * 0.10 +
            confidence.completeness_score * 0.20 +
            confidence.domain_confidence * 0.20 +
            confidence.intent_confidence * 0.15 +
            confidence.knowledge_coverage * 0.15 +
            confidence.conversation_context * 0.10 +
            confidence.metadata_score * 0.05 +
            confidence.retrieval_score * 0.05
        )
        
        return confidence
    
    def _determine_retrieval_strategy(
        self,
        action: ValidationAction,
        domain: Optional[str],
        kb_coverage: float,
        intent: QueryIntent,
    ) -> RetrievalStrategy:
        """Determine appropriate retrieval strategy."""
        if action != ValidationAction.CONTINUE:
            return RetrievalStrategy.NO_RETRIEVAL
        
        if not domain or kb_coverage < 0.3:
            return RetrievalStrategy.NO_RETRIEVAL
        
        if kb_coverage > 0.8:
            return RetrievalStrategy.DIRECT_KB
        
        # Complex intents may need more documents
        complex_intents = {
            QueryIntent.CASE_ANALYSIS,
            QueryIntent.DECISION_SUPPORT,
            QueryIntent.RISK_ANALYSIS,
            QueryIntent.COMPARISON,
            QueryIntent.COMPLAINT,
        }
        
        if intent in complex_intents:
            return RetrievalStrategy.MULTI_DOCUMENT
        
        return RetrievalStrategy.RAG
    
    def _build_response(self, action: ValidationAction, reason: str, domain: Optional[str]) -> str:
        """Build appropriate response."""
        if action == ValidationAction.GREETING:
            return "Hello! I'm your legal assistant. How can I help you with your legal questions today?"
        
        elif action == ValidationAction.GOODBYE:
            return "Thank you for your questions. Feel free to come back if you need more legal assistance. Goodbye!"
        
        elif action == ValidationAction.REJECT:
            if "exceeded" in reason:
                return (
                    "I am here specifically to assist with legal rights and law-related topics. "
                    "If you do not need legal assistance, this chat may not be the right tool. "
                    "Please keep the conversation focused on legal or rights-related questions."
                )
            return "I cannot process this request. Please ask a legal or rights-related question."
        
        elif action == ValidationAction.CLARIFY:
            if "knowledge_gap" in reason:
                return (
                    "I am sorry, but I do not currently have enough verified information about this topic "
                    "in my local legal knowledge base.\n\n"
                    "Currently available areas include:\n\n"
                    "- RTI\n- Consumer Rights\n- Employment Rights\n- Tenant Rights\n"
                    "- Cyber Law and UPI Fraud\n- Contract Law\n- Property, Women, Senior Citizen, and Motor Vehicle topics"
                )
            elif "unsupported" in reason:
                return (
                    f"I do not currently have verified local knowledge for '{domain or 'that topic'}'.\n\n"
                    "Please ask about RTI, tenant rights, consumer rights, employment rights, cyber law, "
                    "UPI fraud, contract clauses, property, women rights, senior citizen rights, or motor vehicle topics."
                )
            elif "incomplete" in reason:
                return (
                    "Could you please complete your question?\n\n"
                    "For example:\n\n"
                    "- What is a contract?\n"
                    "- What is the RTI Act?\n"
                    "- How do I report UPI fraud?\n"
                    "- My landlord refuses to return my security deposit. What should I do?"
                )
            elif "non_legal" in reason:
                return (
                    "I am sorry, but I specialize in legal rights and legal information.\n\n"
                    "I can help with topics like:\n\n"
                    "- Contracts\n- Consumer rights\n- Employment rights\n- RTI\n"
                    "- Cyber law and UPI fraud\n- Tenant and property issues\n\n"
                    "Please ask a legal or rights-related question."
                )
            elif "low_confidence" in reason:
                return (
                    "I am not sure which legal topic you mean yet.\n\n"
                    "Please add more detail, for example:\n\n"
                    "- What is the RTI Act?\n"
                    "- How do I report UPI fraud?\n"
                    "- My employer is not paying salary. What can I do?"
                )
            else:
                return "Could you please provide more details about your legal question? I want to make sure I understand your situation correctly."
        
        return ""
