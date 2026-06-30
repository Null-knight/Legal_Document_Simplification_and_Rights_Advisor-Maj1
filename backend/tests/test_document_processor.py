from app.core.rag.document_processor import DocumentProcessor


def test_document_processor_creates_metadata_chunks():
    processor = DocumentProcessor()
    text = "Section 1 says the tenant must provide written notice. " * 40

    chunks = processor.chunk_text(text, document_id=7, filename="tenant.txt")

    assert chunks
    assert chunks[0]["metadata"]["document_id"] == 7
    assert chunks[0]["metadata"]["filename"] == "tenant.txt"
    assert "text" in chunks[0]
