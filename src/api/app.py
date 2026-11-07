from fastapi import FastAPI

from api.schemas import QueryRequest, QueryResponse
from pipelines.rag_pipeline import RAGPipeline
from retrieval.searcher import Searcher
from storage.repository import InMemoryRepository

app = FastAPI(title="Ingenium RAG API", version="0.1.0")
repository = InMemoryRepository()
pipeline = RAGPipeline(Searcher(repository))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    answer = pipeline.run(request.query, request.top_k)
    return QueryResponse(answer=answer.text, citations=answer.citations)
