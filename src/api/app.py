"""FastAPI service exposing `/health` and `/v1/query` wired to `src/pipeline.py`."""

from functools import lru_cache

from fastapi import Depends, FastAPI

from api.schemas import QueryRequest, QueryResponse
from pipeline import RAGPipeline, build_rag_pipeline

app = FastAPI(title="Ingenium RAG API", version="0.3.0")


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    pipeline, _ = build_rag_pipeline()
    return pipeline


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    answer = pipeline.run(
        request.query,
        top_k=request.top_k,
        project=request.project,
        building=request.building,
        phase=request.phase,
        auto_filter=request.auto_filter,
    )
    return QueryResponse(answer=answer.text, citations=answer.citations)
