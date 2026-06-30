from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = "default"


class SimplifyRequest(BaseModel):
    text: str = Field(..., min_length=1)


class RightsRequest(BaseModel):
    topic: str = Field(..., min_length=1)


class RightsEligibilityRequest(BaseModel):
    domain: str = "tenant"
    facts: dict = {}


class LegalTextRequest(BaseModel):
    text: str = Field(..., min_length=1)


class CompareRequest(BaseModel):
    old_text: str = Field(..., min_length=1)
    new_text: str = Field(..., min_length=1)
