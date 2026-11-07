import hashlib

from ingenium_rag.domain import DocumentMetadata, ParsedDoc, RawFile


class DocumentParser:
    """Dependency-free text/Markdown parser with an adapter boundary for OCR and Docling."""

    def parse(self, raw_file: RawFile) -> ParsedDoc:
        markdown = raw_file.content.decode("utf-8")
        document_id = raw_file.document_id or hashlib.sha256(
            raw_file.source_path.encode()
        ).hexdigest()[:16]
        metadata = DocumentMetadata(
            source_path=raw_file.source_path,
            document_id=document_id,
            extra=raw_file.metadata,
        )
        return ParsedDoc(
            document_id=document_id,
            source_path=raw_file.source_path,
            markdown=markdown,
            metadata=metadata,
        )

    def parse_path(self, path: str) -> ParsedDoc:
        from pathlib import Path

        source = Path(path)
        return self.parse(RawFile(
            document_id=str(source),
            source_path=str(source),
            content=source.read_bytes(),
        ))
