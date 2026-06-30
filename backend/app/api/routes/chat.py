from fastapi import APIRouter, Depends

from app.api.models.request_models import ChatRequest
from app.api.models.response_models import ChatResponse, Citation
from app.core.legal.conversation_controller import ConversationController
from app.core.legal.query_validator import QueryValidator
from app.core.legal.retrieval_pipeline import RetrievalPipeline
from app.db.sqlite import SQLiteRepository
from app.dependencies import get_llm_manager, get_repository, get_retriever


router = APIRouter(prefix="/chat", tags=["chat"])

query_validator = QueryValidator()
retrieval_pipeline = RetrievalPipeline()
conversation_controller = ConversationController(query_validator, retrieval_pipeline)


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    repository: SQLiteRepository = Depends(get_repository),
    retriever=Depends(get_retriever),
    llm_manager=Depends(get_llm_manager),
) -> ChatResponse:
    result = await conversation_controller.process_message(
        session_id=request.session_id,
        message=request.message,
        repository=repository,
        retriever=retriever,
        llm_manager=llm_manager,
    )

    citations = [
        Citation(
            source=citation.get("source", ""),
            content=citation.get("excerpt", ""),
            metadata={
                "title": citation.get("title", ""),
                "reason": citation.get("reason", ""),
            },
            similarity=citation.get("confidence"),
        )
        for citation in result.citations
    ]
    return ChatResponse(answer=result.answer, citations=citations)


@router.get("/history/{session_id}")
def history(session_id: str, repository: SQLiteRepository = Depends(get_repository)) -> dict:
    return {"messages": repository.get_chat_history(session_id)}
