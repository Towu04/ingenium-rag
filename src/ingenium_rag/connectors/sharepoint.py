from collections.abc import Iterator
from pathlib import Path

from ingenium_rag.domain import RawFile


class SharePointConnector:
    """Local export adapter; the Graph API implementation can use the same contract."""

    def __init__(self, site_id: str | None = None) -> None:
        self.site_id = site_id

    def fetch_files(self, path: str | Path) -> Iterator[RawFile]:
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"SharePoint export directory does not exist: {root}")
        for source in root.rglob("*"):
            if source.is_file():
                content = source.read_bytes()
                yield RawFile(
                    document_id=str(source.relative_to(root)),
                    source_path=str(source),
                    content=content,
                )
