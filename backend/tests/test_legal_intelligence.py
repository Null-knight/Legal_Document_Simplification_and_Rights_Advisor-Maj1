from app.core.legal.clause_extractor import ClauseExtractor
from app.core.legal.document_compare import LegalDocumentComparator
from app.core.legal.rights_engine import RightsEligibilityEngine
from app.core.legal.risk_analyzer import ContractRiskAnalyzer


def test_rights_engine_returns_explainable_tenant_rules():
    result = RightsEligibilityEngine().evaluate(
        "tenant",
        {"tenant_months": 14, "eviction_notice_received": True, "deposit_withheld": True},
    )

    applied = [right for right in result["rights"] if right["applies"]]
    assert result["module"] == "Tenant Rights"
    assert any(right["title"] == "Notice before eviction" for right in applied)
    assert all(right["source"] for right in result["rights"])


def test_clause_extractor_detects_legal_clauses():
    text = "This agreement renews automatically unless terminated with 30 days notice. The party shall indemnify the other party."

    clauses = ClauseExtractor().extract(text)

    assert "auto_renewal" in clauses
    assert "indemnity" in clauses


def test_clause_extractor_uses_embeddings_for_cease_language():
    text = "This agreement shall cease after 30 days if either party gives written communication."

    clauses = ClauseExtractor().extract(text)

    assert "termination" in clauses


def test_risk_analyzer_flags_high_risk_terms():
    text = "The agreement shall renew automatically. The vendor shall indemnify and hold harmless the client."

    result = ContractRiskAnalyzer().analyze(text)

    assert any(risk["severity"] == "High" for risk in result["risks"])


def test_document_comparator_reports_changed_notice_period():
    old = "Clause 4. Either party may terminate with 30 days notice."
    new = "Clause 4. Either party may terminate with 7 days notice."

    result = LegalDocumentComparator().compare(old, new)

    assert result["changes"]
    assert "notice" in result["changes"][0]["impact_hint"].lower()
