from functools import lru_cache
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None


load_dotenv()


class Settings:
    APP_NAME = "Legal Document Simplification & Rights Advisor"
    API_PREFIX = "/api"

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
    LEGAL_KNOWLEDGE_BASE_DIR = DATA_DIR / "legal_knowledge_base"
    CHROMA_DIR = DATA_DIR / "chroma_db"
    UPLOAD_DIR = DATA_DIR / "uploads"
    SQLITE_PATH = DATA_DIR / "legal_advisor.sqlite3"

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_FALLBACK_BASE_URL = os.getenv("OLLAMA_FALLBACK_BASE_URL", "http://127.0.0.1:11512")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

    def ensure_directories(self) -> None:
        for path in (
            self.DATA_DIR,
            self.KNOWLEDGE_BASE_DIR,
            self.LEGAL_KNOWLEDGE_BASE_DIR,
            self.CHROMA_DIR,
            self.UPLOAD_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
