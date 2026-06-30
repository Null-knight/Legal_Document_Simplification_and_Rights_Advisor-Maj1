from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from app.config import get_settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteRepository:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or get_settings().SQLITE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    title TEXT,
                    domain TEXT,
                    category TEXT,
                    topics TEXT,
                    source_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    title TEXT,
                    domain TEXT,
                    category TEXT,
                    topics TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            for table, columns in {
                "documents": ["domain", "category", "topics"],
                "document_chunks": ["title", "domain", "category", "topics"],
            }.items():
                for column in columns:
                    self._ensure_column(conn, table, column, "TEXT")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def create_document(
        self,
        filename: str,
        text: str,
        source_type: str = "upload",
        title: str | None = None,
        domain: str | None = None,
        category: str | None = None,
        topics: list[str] | str | None = None,
    ) -> int:
        topics_text = ", ".join(topics) if isinstance(topics, list) else topics
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO documents (filename, title, domain, category, topics, source_type, text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (filename, title or filename, domain, category, topics_text, source_type, text, utc_now()),
            )
            return int(cursor.lastrowid)

    def add_chunks(self, document_id: int, chunks: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO document_chunks (id, document_id, chunk_index, text, title, domain, category, topics, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["id"],
                        document_id,
                        chunk["chunk_index"],
                        chunk["text"],
                        chunk.get("metadata", {}).get("title"),
                        chunk.get("metadata", {}).get("domain"),
                        chunk.get("metadata", {}).get("category"),
                        ", ".join(chunk.get("metadata", {}).get("topics", []))
                        if isinstance(chunk.get("metadata", {}).get("topics"), list)
                        else chunk.get("metadata", {}).get("topics"),
                        utc_now(),
                    )
                    for chunk in chunks
                ],
            )

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT d.id, d.filename, d.title, d.domain, d.category, d.source_type, d.created_at, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_documents_by_source_types(self, source_types: list[str]) -> None:
        if not source_types:
            return
        placeholders = ",".join("?" for _ in source_types)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT id FROM documents WHERE source_type IN ({placeholders})",
                source_types,
            ).fetchall()
            document_ids = [row["id"] for row in rows]
            if document_ids:
                id_placeholders = ",".join("?" for _ in document_ids)
                conn.execute(f"DELETE FROM document_chunks WHERE document_id IN ({id_placeholders})", document_ids)
            conn.execute(f"DELETE FROM documents WHERE source_type IN ({placeholders})", source_types)

    def add_chat_message(self, session_id: str, role: str, content: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, utc_now()),
            )

    def get_chat_history(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            return [dict(row) for row in reversed(rows)]

    def analytics_overview(self) -> dict[str, Any]:
        with self.connect() as conn:
            doc_count = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()["count"]
            chunk_count = conn.execute("SELECT COUNT(*) AS count FROM document_chunks").fetchone()["count"]
            query_count = conn.execute(
                "SELECT COUNT(*) AS count FROM chat_messages WHERE role = 'user'"
            ).fetchone()["count"]
            source_rows = conn.execute(
                "SELECT source_type, COUNT(*) AS count FROM documents GROUP BY source_type ORDER BY count DESC"
            ).fetchall()
            domain_rows = conn.execute(
                """
                SELECT COALESCE(domain, 'unknown') AS domain, COUNT(*) AS count
                FROM documents
                GROUP BY COALESCE(domain, 'unknown')
                ORDER BY count DESC
                """
            ).fetchall()
            question_rows = conn.execute(
                """
                SELECT content, COUNT(*) AS count
                FROM chat_messages
                WHERE role = 'user'
                GROUP BY content
                ORDER BY count DESC
                LIMIT 10
                """
            ).fetchall()
        return {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "total_queries": query_count,
            "documents_by_source": [dict(row) for row in source_rows],
            "documents_by_domain": [dict(row) for row in domain_rows],
            "most_asked_questions": [dict(row) for row in question_rows],
        }
