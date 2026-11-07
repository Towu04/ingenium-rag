from fastapi import FastAPI

from ingenium_rag.api.schemas import QueryRequest, QueryResponse
<<<<<<< HEAD
from ingenium_rag.pipelines.rag_pipeline import RAGPipeline
from ingenium_rag.retrieval.searcher import Searcher
from ingenium_rag.storage.repository import InMemoryRepository

app = FastAPI(title="Ingenium RAG API", version="0.1.0")
repository = InMemoryRepository()
pipeline = RAGPipeline(Searcher(repository))
=======
from ingenium_rag.indexing.vector_store import InMemoryVectorStore
from ingenium_rag.retrieval.generator import GroundedGenerator

app = FastAPI(title="Ingenium RAG API", version="0.1.0")
store = InMemoryVectorStore()
generator = GroundedGenerator()
>>>>>>> d8558de83a60ee17885c0c11a7d88e8a92704c68


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
<<<<<<< HEAD
    answer = pipeline.run(request.query, request.top_k)
=======
    answer = generator.generate(request.query, store.hybrid_search(request.query, request.top_k))
>>>>>>> d8558de83a60ee17885c0c11a7d88e8a92704c68
    return QueryResponse(answer=answer.text, citations=answer.citations)
