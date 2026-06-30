from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents, intelligence, rights
from app.config import get_settings
from app.db.sqlite import SQLiteRepository


settings = get_settings()
SQLiteRepository()

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix=settings.API_PREFIX)
app.include_router(documents.router, prefix=settings.API_PREFIX)
app.include_router(intelligence.router, prefix=settings.API_PREFIX)
app.include_router(rights.router, prefix=settings.API_PREFIX)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": settings.APP_NAME,
        "status": "ready",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
