import hashlib

from domain import RawFile
from processing.chunker import ParentChildChunker
from processing.parser import DocumentParser


def test_parent_child_chunks(sample_markdown, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(sample_markdown, encoding="utf-8")
    content = path.read_bytes()
    document = DocumentParser().parse(RawFile(
        document_id="doc",
        source_path=str(path),
        local_path=path,
        content_hash=hashlib.sha256(content).hexdigest(),
        file_extension=".md",
    ))
    parents, children = ParentChildChunker(100, 25).chunk(document)
    assert parents
    assert children
    assert all(child.parent_id in {parent.id for parent in parents} for child in children)
