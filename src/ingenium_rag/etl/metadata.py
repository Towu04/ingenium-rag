import re
from pathlib import Path
from typing import Any

from ingenium_rag.models import DocumentMetadata


def extract_metadata(source_path: str, rules: dict[str, Any]) -> DocumentMetadata:
    values: dict[str, str | None] = {"project": None, "building": None, "phase": None}
    path = str(Path(source_path))
    for field in values:
        for rule in rules.get(field, []):
            match = re.search(str(rule["pattern"]), path)
            if match:
                values[field] = match.group(int(rule.get("group", 1)))
                break
    return DocumentMetadata(source_path=source_path, **values)
