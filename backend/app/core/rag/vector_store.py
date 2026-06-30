from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.core.rag.embeddings import EmbeddingManager


class VectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embeddings = EmbeddingManager()
        self.client = None
        self.collection = None
        self._available = False
        self._init_chroma()

    def _init_chroma(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(
                path=str(self.settings.CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name="legal_documents",
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
        except Exception:
            self.client = None
            self.collection = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available and self.collection is not None

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks or not self.available:
            return
        texts = [chunk["text"] for chunk in chunks]
        self.collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            documents=texts,
            metadatas=[self._normalize_metadata(chunk["metadata"]) for chunk in chunks],
            embeddings=self.embeddings.embed_documents(texts),
        )

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        normalized: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, list):
                normalized[key] = ", ".join(str(item) for item in value)
            elif isinstance(value, (str, int, float, bool)):
                normalized[key] = value
            else:
                normalized[key] = str(value)
        return normalized

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.available:
            return []
        results = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        matches: list[dict[str, Any]] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for index, document in enumerate(documents):
            distance = distances[index] if index < len(distances) else None
            matches.append(
                {
                    "content": document,
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "similarity": None if distance is None else max(0.0, 1.0 - float(distance)),
                }
            )
        return matches

    def reset_collection(self) -> None:
        if not self.client:
            return
        try:
            self.client.delete_collection("legal_documents")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="legal_documents",
            metadata={"hnsw:space": "cosine"},
        )
        self._available = True

    def stats(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "count": self.collection.count() if self.available else 0,
            "embedding_fallback": self.embeddings.using_fallback,
        }
