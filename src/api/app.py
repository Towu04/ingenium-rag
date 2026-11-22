"""FastAPI service exposing `/health` and `/v1/query` wired to `src/pipeline.py`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI

from api.schemas import QueryRequest, QueryResponse
from pipeline import RAGPipeline, build_rag_pipeline


@lru_cache(maxsize=1)
def _get_pipeline_and_repo() -> tuple[RAGPipeline, Any]:
    return build_rag_pipeline()


def get_pipeline() -> RAGPipeline:
    pipeline, _ = _get_pipeline_and_repo()
    return pipeline


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        if _get_pipeline_and_repo.cache_info().currsize > 0:
            _, repo = _get_pipeline_and_repo()
            repo.close()
            _get_pipeline_and_repo.cache_clear()


_TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Service liveness check.",
    },
    {
        "name": "RAG Query",
        "description": "Hybrid dense + sparse retrieval, cross-encoder reranking, and citation-grounded answering.",
    },
]

app = FastAPI(
    title="Ingenium RAG API",
    description=(
        "Parent-Child Hybrid RAG service for engineering and construction documents "
        "(`BAAI/bge-m3` dense + sparse RRF search, `BAAI/bge-reranker-v2-m3` reranking, "
        "and metadata filtering on `project`, `building`, and `phase`)."
    ),
    version="0.3.0",
    openapi_tags=_TAGS_METADATA,
    lifespan=_lifespan,
)


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    description="Returns `{status: 'ok'}` when the FastAPI service is running.",
)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/query",
    response_model=QueryResponse,
    tags=["RAG Query"],
    summary="Query the RAG index",
    description=(
        "Runs hybrid dense + sparse RRF retrieval over local Qdrant, reranks candidate "
        "parent chunks with BGE-M3, and returns a grounded answer with document citations."
    ),
)
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
