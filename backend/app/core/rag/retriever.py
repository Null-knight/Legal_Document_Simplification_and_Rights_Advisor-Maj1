from __future__ import annotations

import re
from typing import Any

from app.core.rag.vector_store import VectorStore
from app.db.sqlite import SQLiteRepository


class Retriever:
    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.repository = SQLiteRepository()

    def retrieve(self, query: str, top_k: int = 5, domain: str | None = None) -> list[dict[str, Any]]:
        semantic = self.vector_store.search(query, top_k=top_k * 4)
        lexical = self._sqlite_bm25_search(query, top_k=top_k * 4, domain=domain)
        combined = self._combine_scores(semantic, lexical)
        if domain:
            combined = [
                item for item in combined if item.get("metadata", {}).get("domain", "").lower() == domain.lower()
            ]
        return self._rerank(query, combined)[:top_k]

    def _sqlite_bm25_search(self, query: str, top_k: int, domain: str | None = None) -> list[dict[str, Any]]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        with self.repository.connect() as conn:
            where = "WHERE UPPER(c.domain) = ?" if domain else ""
            params = [domain.upper()] if domain else []
            rows = conn.execute(
                f"""
                SELECT c.text, c.chunk_index, c.title, c.domain, c.category, c.topics, d.id AS document_id, d.filename
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                {where}
                """
                ,
                params,
            ).fetchall()
        if not rows:
            return []
        corpus = [self._tokenize(row["text"]) for row in rows]
        scores = self._bm25_scores(corpus, query_terms)
        max_score = max(scores) if scores else 0.0
        scored = []
        for row, score in zip(rows, scores):
            if score <= 0:
                continue
            normalized = score / max_score if max_score else 0.0
            scored.append(
                {
                    "content": row["text"],
                    "metadata": {
                        "document_id": row["document_id"],
                        "filename": row["filename"],
                        "chunk_index": row["chunk_index"],
                        "title": row["title"],
                        "domain": row["domain"],
                        "category": row["category"],
                        "topics": row["topics"],
                    },
                    "similarity": normalized,
                    "bm25_score": normalized,
                }
            )
        return sorted(scored, key=lambda item: item["bm25_score"], reverse=True)[:top_k]

    def _bm25_scores(self, corpus: list[list[str]], query_terms: list[str]) -> list[float]:
        try:
            from rank_bm25 import BM25Okapi

            return [float(score) for score in BM25Okapi(corpus).get_scores(query_terms)]
        except Exception:
            scores = []
            for tokens in corpus:
                token_set = set(tokens)
                scores.append(float(sum(1 for term in query_terms if term in token_set)))
            return scores

    def _combine_scores(self, semantic: list[dict[str, Any]], lexical: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_key: dict[str, dict[str, Any]] = {}
        for item in semantic:
            key = self._key(item)
            item = dict(item)
            item["vector_score"] = item.get("similarity") or 0.0
            item["bm25_score"] = 0.0
            by_key[key] = item
        for item in lexical:
            key = self._key(item)
            existing = by_key.get(key, dict(item))
            existing["bm25_score"] = max(existing.get("bm25_score", 0.0), item.get("bm25_score", 0.0))
            existing["vector_score"] = existing.get("vector_score", 0.0)
            existing["metadata"] = {**item.get("metadata", {}), **existing.get("metadata", {})}
            by_key[key] = existing
        for item in by_key.values():
            vector_score = item.get("vector_score", 0.0)
            bm25_score = item.get("bm25_score", 0.0)
            combined = (0.65 * vector_score) + (0.35 * bm25_score)
            item["similarity"] = combined
            item["confidence"] = round(min(0.99, combined), 3)
        return sorted(by_key.values(), key=lambda item: item["similarity"], reverse=True)

    def _rerank(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_terms = set(self._tokenize(query))
        for item in results:
            metadata_text = " ".join(str(value or "") for value in item.get("metadata", {}).values())
            metadata_overlap = len(query_terms & set(self._tokenize(metadata_text)))
            item["rerank_score"] = item.get("similarity", 0.0) + (0.03 * metadata_overlap)
        return sorted(results, key=lambda item: item["rerank_score"], reverse=True)

    def _key(self, item: dict[str, Any]) -> str:
        metadata = item.get("metadata", {})
        return f"{metadata.get('document_id')}:{metadata.get('chunk_index')}:{hash(item.get('content', '')[:120])}"

    def _tokenize(self, text: str) -> list[str]:
        return [term for term in re.findall(r"\w+", text.lower()) if len(term) > 2]

    def stats(self) -> dict[str, Any]:
        return self.vector_store.stats()
