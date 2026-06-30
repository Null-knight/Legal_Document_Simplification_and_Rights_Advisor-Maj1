from typing import Any

from pydantic import BaseModel


class Citation(BaseModel):
    source: str
    content: str
    metadata: dict[str, Any] = {}
    similarity: float | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []


class UploadResponse(BaseModel):
    document_id: int
    filename: str
    chunks_indexed: int


class DocumentListItem(BaseModel):
    id: int
    filename: str
    title: str | None
    domain: str | None = None
    category: str | None = None
    source_type: str
    created_at: str
    chunk_count: int
