from __future__ import annotations

from hashlib import sha1
import re

from app.config import get_settings
from app.core.utils.text_cleaner import clean_text


class DocumentProcessor:
    def __init__(self) -> None:
        self.settings = get_settings()

    def chunk_text(self, text: str, document_id: int, filename: str, base_metadata: dict | None = None) -> list[dict]:
        cleaned = clean_text(text)
        if not cleaned:
            return []

        chunks: list[dict] = []
        start = 0
        chunk_index = 0
        while start < len(cleaned):
            end = min(start + self.settings.CHUNK_SIZE, len(cleaned))
            chunk = self._trim_to_sentence(cleaned[start:end])
            if chunk:
                chunk_id = sha1(f"{document_id}:{chunk_index}:{chunk}".encode("utf-8")).hexdigest()
                metadata = {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "has_law_citation": bool(re.search(r"\b(section|article|act|rule)\b", chunk, re.I)),
                }
                if base_metadata:
                    metadata.update(base_metadata)
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk,
                        "chunk_index": chunk_index,
                        "metadata": metadata,
                    }
                )
                chunk_index += 1
            if end == len(cleaned):
                break
            start = max(end - self.settings.CHUNK_OVERLAP, start + 1)
        return chunks

    def _trim_to_sentence(self, text: str) -> str:
        text = text.strip()
        if len(text) < 200:
            return text
        match = list(re.finditer(r"[.!?]\s+", text))
        if match:
            last = match[-1].end()
            if last > len(text) * 0.55:
                return text[:last].strip()
        return text
