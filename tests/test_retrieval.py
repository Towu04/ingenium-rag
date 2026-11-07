<<<<<<< HEAD
from ingenium_rag.domain import ChildChunk, DocumentMetadata, SearchResult
from ingenium_rag.retrieval.reranker import Reranker
=======
from ingenium_rag.models import ChildChunk, DocumentMetadata, SearchResult
from ingenium_rag.retrieval.search import filter_results
>>>>>>> d8558de83a60ee17885c0c11a7d88e8a92704c68


def test_metadata_filter():
    metadata = DocumentMetadata(source_path="a.md", project="A")
    chunk = ChildChunk(id="c", parent_id="p", document_id="d", text="answer", metadata=metadata)
<<<<<<< HEAD
    result = SearchResult(chunk=chunk, score=1)
    assert Reranker().rerank("answer", [result])
=======
    assert filter_results([SearchResult(chunk=chunk, score=1)], {"project": "A"})
    assert not filter_results([SearchResult(chunk=chunk, score=1)], {"project": "B"})
>>>>>>> d8558de83a60ee17885c0c11a7d88e8a92704c68
