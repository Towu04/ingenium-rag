"""FastAPI service exposing `/health` and `/v1/query` with lazy pipeline initialization."""

from functools import lru_cache

from fastapi import Depends, FastAPI

from api.schemas import QueryRequest, QueryResponse
from config import get_settings
from generation.generator import ExtractiveStubLLM, GroundedGenerator, LLMClient, OpenAICompatibleLLM
from pipelines.rag_pipeline import RAGPipeline
from retrieval.reranker import BGEM3Reranker
from retrieval.searcher import Searcher
from storage.embedder import BGEM3Embedder
from storage.repository import LocalHybridRepository

app = FastAPI(title="Ingenium RAG API", version="0.2.0")


def build_default_pipeline() -> RAGPipeline:
    """Instantiate the local Qdrant + BGE-M3 + reranker + generator stack from `Settings`."""
    settings = get_settings()
    repository = LocalHybridRepository(path=str(settings.qdrant_path))
    embedder = BGEM3Embedder(model_name=settings.embedding_model, device=settings.device)
    reranker = BGEM3Reranker(model_name=settings.reranker_model, device=settings.device)

    llm: LLMClient
    if settings.llm_base_url and settings.llm_model:
        llm = OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
        )
    else:
        llm = ExtractiveStubLLM()

    return RAGPipeline(
        searcher=Searcher(embedder=embedder, repository=repository),
        reranker=reranker,
        generator=GroundedGenerator(llm=llm),
        retrieve_k=settings.retrieve_k,
        child_candidate_limit=settings.child_candidate_limit,
        rerank_k=settings.rerank_k,
    )


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return build_default_pipeline()


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
