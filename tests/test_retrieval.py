from domain import DocumentMetadata, ParentChunk, SearchResult
from retrieval.reranker import Reranker


def test_metadata_filter():
    metadata = DocumentMetadata(source_path="a.md", project="A")
    parent = ParentChunk(
        id="p",
        document_id="d",
        parent_index=0,
        text="answer",
        metadata=metadata,
    )
    result = SearchResult(parent_chunk=parent, matched_child_id="c", score=1)
    assert Reranker().rerank("answer", [result]) == [result]
