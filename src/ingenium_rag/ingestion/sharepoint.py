from collections.abc import Iterator
from pathlib import Path


class SharePointClient:
    """Adapter boundary for Microsoft Graph; local iteration is useful for development."""

    def __init__(self, site_id: str | None = None) -> None:
        self.site_id = site_id

    def iter_files(self, path: str | Path) -> Iterator[Path]:
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"SharePoint export directory does not exist: {root}")
        yield from (file for file in root.rglob("*") if file.is_file())
