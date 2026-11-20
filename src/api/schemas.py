from pydantic import BaseModel, Field

from domain import Citation


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    project: str | None = None
    building: str | None = None
    phase: str | None = None
    auto_filter: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
