from ingenium_rag.etl.parser import DocumentParser
from ingenium_rag.indexing.chunker import ParentChildChunker


def test_parent_child_chunks(sample_markdown, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(sample_markdown, encoding="utf-8")
    document = DocumentParser().parse(path)
    parents, children = ParentChildChunker(100, 25).chunk(document)
    assert parents
    assert children
    assert all(child.parent_id in {parent.id for parent in parents} for child in children)
