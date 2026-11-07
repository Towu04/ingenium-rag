import hashlib
from pathlib import Path

from domain import DocumentMetadata, ParsedDoc, ParsedSection, RawFile

RawFile.model_rebuild(_types_namespace={"Path": Path})


class DocumentParser:
    """Dependency-free text/Markdown parser with an adapter boundary for OCR and Docling."""

    def parse(self, raw_file: RawFile) -> ParsedDoc:
        content = raw_file.local_path.read_bytes()
        markdown = content.decode("utf-8")
        document_id = raw_file.document_id or hashlib.sha256(
            raw_file.source_path.encode()
        ).hexdigest()[:16]
        metadata = DocumentMetadata(
            source_path=raw_file.source_path,
            sharepoint_url=raw_file.sharepoint_url,
        )
        return ParsedDoc(
            document_id=document_id,
            content_hash=raw_file.content_hash,
            markdown=markdown,
            sections=[ParsedSection(text=markdown)],
            metadata=metadata,
        )

    def parse_path(self, path: str) -> ParsedDoc:
        source = Path(path)
        content = source.read_bytes()
        return self.parse(RawFile(
            document_id=str(source),
            source_path=str(source),
            local_path=source,
            content_hash=hashlib.sha256(content).hexdigest(),
            file_extension=source.suffix.lower(),
        ))
