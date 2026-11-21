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


app = FastAPI(title="Ingenium RAG API", version="0.3.0", lifespan=_lifespan)


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
