import re


def extract_metadata_filters(query: str) -> dict[str, str]:
    filters: dict[str, str] = {}
    for field in ("building", "phase", "project"):
        match = re.search(rf"\b{field}\s*[:=]?\s*([A-Za-z0-9_-]+)", query, re.IGNORECASE)
        if match:
            filters[field] = match.group(1)
    return filters
