<<<<<<< HEAD
from ingenium_rag.domain import RawFile
from ingenium_rag.processing.chunker import ParentChildChunker
from ingenium_rag.processing.parser import DocumentParser
=======
from ingenium_rag.etl.parser import DocumentParser
from ingenium_rag.indexing.chunker import ParentChildChunker
>>>>>>> d8558de83a60ee17885c0c11a7d88e8a92704c68


def test_parent_child_chunks(sample_markdown, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(sample_markdown, encoding="utf-8")
<<<<<<< HEAD
    document = DocumentParser().parse(RawFile(
        document_id="doc",
        source_path=str(path),
        content=path.read_bytes(),
    ))
=======
    document = DocumentParser().parse(path)
>>>>>>> d8558de83a60ee17885c0c11a7d88e8a92704c68
    parents, children = ParentChildChunker(100, 25).chunk(document)
    assert parents
    assert children
    assert all(child.parent_id in {parent.id for parent in parents} for child in children)
