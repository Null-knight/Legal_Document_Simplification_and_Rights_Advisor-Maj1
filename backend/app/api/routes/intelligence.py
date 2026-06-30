import httpx
from fastapi import APIRouter, Depends

from app.api.models.request_models import CompareRequest, LegalTextRequest, RightsEligibilityRequest
from app.config import get_settings
from app.core.legal.clause_extractor import ClauseExtractor
from app.core.legal.document_compare import LegalDocumentComparator
from app.core.legal.rights_engine import RIGHT_MODULES, RightsEligibilityEngine
from app.core.legal.risk_analyzer import ContractRiskAnalyzer
from app.db.sqlite import SQLiteRepository
from app.dependencies import get_repository, get_retriever


router = APIRouter(prefix="/intelligence", tags=["legal intelligence"])


@router.get("/modules")
def modules() -> dict:
    return {"modules": RIGHT_MODULES}


@router.post("/rights-eligibility")
def rights_eligibility(request: RightsEligibilityRequest, retriever=Depends(get_retriever)) -> dict:
    result = RightsEligibilityEngine().evaluate(request.domain, request.facts)
    for right in result["rights"]:
        if not right["applies"]:
            continue
        query = f"{result['module']} {right['title']} {right['explanation']}"
        evidence = retriever.retrieve(query, top_k=2, domain=result["domain"])
        right["supporting_sources"] = [
            {
                "title": item.get("metadata", {}).get("title") or item.get("metadata", {}).get("filename"),
                "filename": item.get("metadata", {}).get("filename"),
                "confidence": item.get("confidence", item.get("similarity")),
                "excerpt": item.get("content", "")[:260],
            }
            for item in evidence
        ]
        if evidence:
            result["confidence"] = round(min(0.99, result["confidence"] + 0.08), 2)
    return result


@router.post("/extract-clauses")
def extract_clauses(request: LegalTextRequest) -> dict:
    return {"clauses": ClauseExtractor().extract(request.text)}


@router.post("/analyze-risk")
def analyze_risk(request: LegalTextRequest) -> dict:
    return ContractRiskAnalyzer().analyze(request.text)


@router.post("/compare")
def compare_documents(request: CompareRequest) -> dict:
    return LegalDocumentComparator().compare(request.old_text, request.new_text)


@router.get("/analytics")
def analytics(repository: SQLiteRepository = Depends(get_repository)) -> dict:
    return repository.analytics_overview()


@router.get("/knowledge-graph")
def knowledge_graph() -> dict:
    nodes = [
        {"id": "contract", "label": "Contract Law"},
        {"id": "tenant", "label": "Tenant Rights"},
        {"id": "consumer", "label": "Consumer Rights"},
        {"id": "employment", "label": "Employment Rights"},
        {"id": "cyber", "label": "Cyber Rights"},
        {"id": "women", "label": "Women Protection"},
        {"id": "rti", "label": "RTI"},
        {"id": "property", "label": "Property Rights"},
        {"id": "senior", "label": "Senior Citizen Rights"},
    ]
    edges = [
        {"source": "contract", "target": "tenant", "relation": "lease agreements"},
        {"source": "contract", "target": "consumer", "relation": "purchase and warranty terms"},
        {"source": "contract", "target": "employment", "relation": "employment agreements"},
        {"source": "property", "target": "tenant", "relation": "rental housing"},
        {"source": "cyber", "target": "women", "relation": "online harassment"},
        {"source": "rti", "target": "consumer", "relation": "public information requests"},
        {"source": "senior", "target": "property", "relation": "property pressure and protection"},
    ]
    return {"nodes": nodes, "edges": edges}


@router.get("/system-status")
def system_status(repository: SQLiteRepository = Depends(get_repository)) -> dict:
    settings = get_settings()
    ollama_connected = False
    ollama_urls = [
        settings.OLLAMA_BASE_URL.rstrip("/"),
        settings.OLLAMA_FALLBACK_BASE_URL.rstrip("/"),
    ]
    for base_url in dict.fromkeys(ollama_urls):
        try:
            response = httpx.get(f"{base_url}/api/tags", timeout=1.5)
            if response.status_code == 200:
                ollama_connected = True
                break
        except Exception:
            continue
    analytics = repository.analytics_overview()
    vector_stats = _chroma_stats(settings)
    return {
        "ollama_connected": ollama_connected,
        "chromadb_active": bool(vector_stats.get("available")),
        "rights_engine_loaded": True,
        "knowledge_base_ready": analytics["total_chunks"] > 0,
        "documents_indexed": analytics["total_documents"],
        "chunks_indexed": analytics["total_chunks"],
        "embedding_fallback": vector_stats.get("embedding_fallback", True),
    }


def _chroma_stats(settings) -> dict:
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            name="legal_documents",
            metadata={"hnsw:space": "cosine"},
        )
        return {
            "available": True,
            "count": collection.count(),
            "embedding_fallback": False,
        }
    except Exception:
        return {
            "available": False,
            "count": 0,
            "embedding_fallback": True,
        }
