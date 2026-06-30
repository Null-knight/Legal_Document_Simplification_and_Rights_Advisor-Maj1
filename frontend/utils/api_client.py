from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


class ApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/health", timeout=3).ok
        except requests.RequestException:
            return False

    def upload_document(self, path: Path) -> dict[str, Any]:
        with path.open("rb") as file:
            response = requests.post(
                f"{self.base_url}/api/documents/upload",
                files={"file": (path.name, file)},
                timeout=120,
            )
        response.raise_for_status()
        return response.json()

    def simplify(self, text: str) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/api/documents/simplify", json={"text": text}, timeout=120)
        response.raise_for_status()
        return response.json()

    def chat(self, message: str, session_id: str = "streamlit") -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"message": message, "session_id": session_id},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def rights(self, topic: str) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/api/rights", json={"topic": topic}, timeout=120)
        response.raise_for_status()
        return response.json()

    def documents(self) -> list[dict[str, Any]]:
        response = requests.get(f"{self.base_url}/api/documents", timeout=30)
        response.raise_for_status()
        return response.json()

    def rights_modules(self) -> dict[str, str]:
        response = requests.get(f"{self.base_url}/api/intelligence/modules", timeout=30)
        response.raise_for_status()
        return response.json()["modules"]

    def rights_eligibility(self, domain: str, facts: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/intelligence/rights-eligibility",
            json={"domain": domain, "facts": facts},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def extract_clauses(self, text: str) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/api/intelligence/extract-clauses", json={"text": text}, timeout=60)
        response.raise_for_status()
        return response.json()

    def analyze_risk(self, text: str) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/api/intelligence/analyze-risk", json={"text": text}, timeout=60)
        response.raise_for_status()
        return response.json()

    def compare_documents(self, old_text: str, new_text: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/intelligence/compare",
            json={"old_text": old_text, "new_text": new_text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def analytics(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/api/intelligence/analytics", timeout=30)
        response.raise_for_status()
        return response.json()

    def system_status(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/api/intelligence/system-status", timeout=30)
        response.raise_for_status()
        return response.json()
