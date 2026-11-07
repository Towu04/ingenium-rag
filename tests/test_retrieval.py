from ingenium_rag.models import ChildChunk, DocumentMetadata, SearchResult
from ingenium_rag.retrieval.search import filter_results


def test_metadata_filter():
    metadata = DocumentMetadata(source_path="a.md", project="A")
    chunk = ChildChunk(id="c", parent_id="p", document_id="d", text="answer", metadata=metadata)
    assert filter_results([SearchResult(chunk=chunk, score=1)], {"project": "A"})
    assert not filter_results([SearchResult(chunk=chunk, score=1)], {"project": "B"})
