from __future__ import annotations

from hashlib import blake2b
import math


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", fallback_dimension: int = 384) -> None:
        self.model_name = model_name
        self.fallback_dimension = fallback_dimension
        self._model = None
        self._load_error: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            self._load_error = str(exc)
            self._model = None

    @property
    def using_fallback(self) -> bool:
        return self._model is None

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [embedding.tolist() for embedding in embeddings]
        return [self._hash_embedding(text) for text in texts]

    def _hash_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.fallback_dimension
        for token in text.lower().split():
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.fallback_dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
