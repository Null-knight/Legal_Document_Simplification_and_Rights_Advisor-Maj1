from functools import lru_cache

from app.core.llm.llm_manager import LLMManager
from app.core.rag.retriever import Retriever
from app.db.sqlite import SQLiteRepository


@lru_cache
def get_repository() -> SQLiteRepository:
    return SQLiteRepository()


@lru_cache
def get_retriever() -> Retriever:
    return Retriever()


@lru_cache
def get_llm_manager() -> LLMManager:
    return LLMManager()
