from fastapi import FastAPI

from ingenium_rag.api.schemas import QueryRequest, QueryResponse
from ingenium_rag.indexing.vector_store import InMemoryVectorStore
from ingenium_rag.retrieval.generator import GroundedGenerator

app = FastAPI(title="Ingenium RAG API", version="0.1.0")
store = InMemoryVectorStore()
generator = GroundedGenerator()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    answer = generator.generate(request.query, store.hybrid_search(request.query, request.top_k))
    return QueryResponse(answer=answer.text, citations=answer.citations)
