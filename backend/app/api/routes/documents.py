from pathlib import Path
import shutil

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.models.request_models import SimplifyRequest
from app.api.models.response_models import DocumentListItem, UploadResponse
from app.config import get_settings
from app.core.rag.document_processor import DocumentProcessor
from app.core.rag.retriever import Retriever
from app.core.simplification.simplifier import LegalSimplifier
from app.core.utils.file_handler import extract_text
from app.db.sqlite import SQLiteRepository
from app.dependencies import get_llm_manager, get_repository, get_retriever


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    repository: SQLiteRepository = Depends(get_repository),
    retriever: Retriever = Depends(get_retriever),
) -> UploadResponse:
    settings = get_settings()
    destination = settings.UPLOAD_DIR / Path(file.filename or "document.txt").name
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = extract_text(destination)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document_id = repository.create_document(destination.name, text, source_type="upload")
    chunks = DocumentProcessor().chunk_text(text, document_id=document_id, filename=destination.name)
    repository.add_chunks(document_id, chunks)
    retriever.vector_store.add_chunks(chunks)
    return UploadResponse(document_id=document_id, filename=destination.name, chunks_indexed=len(chunks))


@router.post("/simplify")
async def simplify_text(
    request: SimplifyRequest,
    llm_manager=Depends(get_llm_manager),
) -> dict[str, str]:
    simplifier = LegalSimplifier(llm_manager)
    return {"summary": await simplifier.simplify(request.text)}


@router.get("", response_model=list[DocumentListItem])
def list_documents(repository: SQLiteRepository = Depends(get_repository)) -> list[dict]:
    return repository.list_documents()


@router.get("/search")
def search_documents(query: str, domain: str | None = None, retriever: Retriever = Depends(get_retriever)) -> dict:
    return {"results": retriever.retrieve(query, domain=domain), "stats": retriever.stats()}
