from app.db.sqlite import SQLiteRepository


def test_repository_stores_documents_chunks_and_chat(tmp_path):
    repository = SQLiteRepository(tmp_path / "test.sqlite3")

    document_id = repository.create_document("lease.txt", "Tenant must give notice.")
    repository.add_chunks(
        document_id,
        [{"id": "chunk-1", "chunk_index": 0, "text": "Tenant must give notice."}],
    )
    repository.add_chat_message("session-1", "user", "What notice is needed?")

    documents = repository.list_documents()
    history = repository.get_chat_history("session-1")

    assert documents[0]["filename"] == "lease.txt"
    assert documents[0]["chunk_count"] == 1
    assert history[0]["content"] == "What notice is needed?"
