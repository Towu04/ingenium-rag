import re
from typing import Any

from domain import DocumentMetadata, ParsedDoc


def extract_metadata(source_path: str, rules: dict[str, Any]) -> DocumentMetadata:
    values: dict[str, str | None] = {"project": None, "building": None, "phase": None}
    for field in values:
        for rule in rules.get(field, []):
            match = re.search(str(rule["pattern"]), source_path)
            if match:
                values[field] = match.group(int(rule.get("group", 1)))
                break
    return DocumentMetadata(source_path=source_path, **values)


def apply_metadata(document: ParsedDoc, rules: dict[str, Any]) -> ParsedDoc:
    metadata = extract_metadata(document.metadata.source_path, rules)
    return document.model_copy(update={"metadata": metadata})
