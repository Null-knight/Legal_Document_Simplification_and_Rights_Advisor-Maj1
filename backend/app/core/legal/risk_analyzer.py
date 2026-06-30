from __future__ import annotations

import re

from app.core.legal.clause_extractor import ClauseExtractor


RISK_RULES = {
    "auto_renewal": ("High", "Auto-renewal can lock a party into a new term unless notice is given in time."),
    "penalty": ("Medium", "Penalty or forfeiture clauses may create financial exposure."),
    "indemnity": ("High", "Broad indemnity can shift large losses or third-party claims to one party."),
    "liability": ("Medium", "Liability language should be checked for caps, exclusions, and unlimited exposure."),
    "termination": ("Medium", "Termination clauses affect exit rights, notice periods, and breach consequences."),
    "arbitration": ("Low", "Arbitration changes the dispute forum and procedure."),
    "governing_law": ("Low", "Jurisdiction clauses determine where disputes may be heard."),
    "confidentiality": ("Low", "Confidentiality clauses can create continuing obligations after the contract ends."),
}


class ContractRiskAnalyzer:
    def __init__(self) -> None:
        self.extractor = ClauseExtractor()

    def analyze(self, text: str) -> dict[str, list[dict[str, str]]]:
        clauses = self.extractor.extract(text)
        risks = []
        for clause_type, detected_clauses in clauses.items():
            if clause_type not in RISK_RULES:
                continue
            severity, explanation = RISK_RULES[clause_type]
            for detected in detected_clauses:
                severity, explanation = self._adjust_severity(clause_type, detected["clause"], severity, explanation)
                risks.append(
                    {
                        "risk_type": clause_type,
                        "severity": severity,
                        "explanation": explanation,
                        "clause": detected["clause"],
                        "source": "Rule-based contract risk analyzer",
                    }
                )
        severity_order = {"High": 0, "Medium": 1, "Low": 2}
        return {
            "risks": sorted(risks, key=lambda item: severity_order[item["severity"]]),
            "clause_summary": {key: len(value) for key, value in clauses.items()},
        }

    def _adjust_severity(self, clause_type: str, clause: str, severity: str, explanation: str) -> tuple[str, str]:
        lowered = clause.lower()
        if clause_type == "liability":
            has_cap = bool(re.search(r"\b(cap|capped|limited to|maximum liability|not exceed)\b", lowered))
            unlimited = bool(re.search(r"\b(unlimited|without limit|all losses|any and all)\b", lowered))
            if unlimited:
                return "High", "Liability appears broad or unlimited, which can create major exposure."
            if has_cap:
                return "Low", "A liability cap appears present; verify whether exclusions or uncapped items remain."
            return "High", "No clear liability cap was detected, so exposure may be high."
        if clause_type == "termination" and not re.search(r"\bnotice|days|months|written\b", lowered):
            return "High", "Termination language was detected without a clear notice period or process."
        if clause_type == "auto_renewal" and re.search(r"\bunless terminated|without notice|successive\b", lowered):
            return "High", explanation
        return severity, explanation
