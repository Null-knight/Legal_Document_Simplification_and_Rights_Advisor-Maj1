from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.config import get_settings
from app.core.rag.document_processor import DocumentProcessor
from app.core.rag.retriever import Retriever
from app.core.utils.file_handler import extract_text
from app.db.sqlite import SQLiteRepository


DOMAIN_BY_FOLDER = {
    "tenant": "TENANT",
    "consumer": "CONSUMER",
    "employment": "EMPLOYMENT",
    "cyber": "CYBER",
    "women": "WOMEN",
    "rti": "RTI",
    "property": "PROPERTY",
    "senior": "SENIOR",
    "motor_vehicle": "MOTOR_VEHICLE",
    "contract": "CONTRACT",
}


def parse_metadata(text: str, path: Path) -> dict:
    metadata = {
        "title": path.stem.replace("_", " ").title(),
        "domain": DOMAIN_BY_FOLDER.get(path.parent.name, path.parent.name.upper()),
        "category": path.parent.name.replace("_", " ").title(),
        "topics": [],
    }
    for line in text.splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "title" and value:
            metadata["title"] = value
        elif key == "domain" and value:
            metadata["domain"] = value.upper()
        elif key == "category" and value:
            metadata["category"] = value
        elif key == "topics" and value:
            metadata["topics"] = [item.strip() for item in value.split(",") if item.strip()]
    if not metadata["topics"]:
        metadata["topics"] = [item.strip() for item in metadata["title"].lower().split() if item.strip()]
    return metadata


def main() -> None:
    settings = get_settings()
    repository = SQLiteRepository()
    retriever = Retriever()
    processor = DocumentProcessor()
    reset = "--reset" in sys.argv

    if reset:
        repository.delete_documents_by_source_types(["knowledge_base", "legal_knowledge_base"])
        retriever.vector_store.reset_collection()

    total_chunks = 0
    source_dirs = [settings.KNOWLEDGE_BASE_DIR, settings.LEGAL_KNOWLEDGE_BASE_DIR]
    for path in [file for source_dir in source_dirs for file in source_dir.rglob("*")]:
        if path.suffix.lower() not in {".txt", ".pdf", ".docx"}:
            continue
        text = extract_text(path)
        source_type = "legal_knowledge_base" if path.parent == settings.LEGAL_KNOWLEDGE_BASE_DIR else "knowledge_base"
        if settings.LEGAL_KNOWLEDGE_BASE_DIR in path.parents:
            source_type = "legal_knowledge_base"
        metadata = parse_metadata(text, path)
        document_id = repository.create_document(
            path.name,
            text,
            source_type=source_type,
            title=metadata["title"],
            domain=metadata["domain"],
            category=metadata["category"],
            topics=metadata["topics"],
        )
        chunks = processor.chunk_text(text, document_id=document_id, filename=path.name, base_metadata=metadata)
        repository.add_chunks(document_id, chunks)
        retriever.vector_store.add_chunks(chunks)
        total_chunks += len(chunks)
        print(f"Indexed {metadata['domain']} / {path.name}: {len(chunks)} chunks")
    print(f"Done. Indexed {total_chunks} chunks.")


if __name__ == "__main__":
    main()
