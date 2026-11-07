from pydantic import BaseModel, Field

<<<<<<< HEAD
from ingenium_rag.domain import Citation
=======
from ingenium_rag.models import Citation
>>>>>>> d8558de83a60ee17885c0c11a7d88e8a92704c68


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
