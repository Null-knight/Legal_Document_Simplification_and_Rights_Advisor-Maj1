from __future__ import annotations

import httpx

from app.config import get_settings


class LLMManager:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def generate(self, prompt: str) -> str:
        if self.settings.LLM_PROVIDER == "groq" and self.settings.GROQ_API_KEY:
            return await self._generate_groq(prompt)
        return await self._generate_ollama(prompt)

    async def _generate_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        for base_url in self._ollama_base_urls():
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=1.5)) as client:
                    response = await client.post(f"{base_url}/api/generate", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data.get("response", "").strip()
            except Exception:
                continue
        return ""

    def _ollama_base_urls(self) -> list[str]:
        urls = [
            self.settings.OLLAMA_BASE_URL.rstrip("/"),
            self.settings.OLLAMA_FALLBACK_BASE_URL.rstrip("/"),
        ]
        return list(dict.fromkeys(urls))

    async def _generate_groq(self, prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.GROQ_API_KEY}"}
        payload = {
            "model": self.settings.GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""
