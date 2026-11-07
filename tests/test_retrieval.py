from domain import ChildChunk, DocumentMetadata, SearchResult
from retrieval.reranker import Reranker


def test_metadata_filter():
    metadata = DocumentMetadata(source_path="a.md", project="A")
    chunk = ChildChunk(id="c", parent_id="p", document_id="d", text="answer", metadata=metadata)
    result = SearchResult(chunk=chunk, score=1)
    assert Reranker().rerank("answer", [result])
