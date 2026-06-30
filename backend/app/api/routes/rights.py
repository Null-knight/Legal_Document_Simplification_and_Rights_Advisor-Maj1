from fastapi import APIRouter, Depends

from app.api.models.request_models import RightsRequest
from app.api.models.response_models import ChatResponse, Citation
from app.core.llm.prompts import LEGAL_DISCLAIMER, RIGHTS_PROMPT
from app.dependencies import get_llm_manager, get_retriever


router = APIRouter(prefix="/rights", tags=["rights"])


@router.post("", response_model=ChatResponse)
async def explain_rights(
    request: RightsRequest,
    retriever=Depends(get_retriever),
    llm_manager=Depends(get_llm_manager),
) -> ChatResponse:
    matches = retriever.retrieve(request.topic)
    context = "\n\n".join(f"[Source {index + 1}] {item['content']}" for index, item in enumerate(matches))
    answer = ""
    if context:
        answer = await llm_manager.generate(RIGHTS_PROMPT.format(topic=request.topic, context=context))
    if not answer:
        answer = (
            "I can explain rights from documents you add to the local knowledge base. "
            "Right now I do not have enough matching source text for this topic.\n\n"
            f"{LEGAL_DISCLAIMER}"
        )
    citations = [
        Citation(
            source=f"Source {index + 1}",
            content=item["content"][:350],
            metadata=item.get("metadata", {}),
            similarity=item.get("similarity"),
        )
        for index, item in enumerate(matches)
    ]
    return ChatResponse(answer=answer, citations=citations)
