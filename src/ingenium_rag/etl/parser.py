import hashlib
from pathlib import Path

from ingenium_rag.models import DocumentMetadata, ParsedDoc


class DocumentParser:
    """Dependency-free Markdown/text parser; richer Docling/OCR adapters can replace it."""

    def parse(self, path: str | Path) -> ParsedDoc:
        source = Path(path)
        content = source.read_text(encoding="utf-8")
        document_id = hashlib.sha256(str(source).encode()).hexdigest()[:16]
        metadata = DocumentMetadata(source_path=str(source), document_id=document_id)
        return ParsedDoc(
            document_id=document_id,
            source_path=str(source),
            markdown=content,
            metadata=metadata,
        )
