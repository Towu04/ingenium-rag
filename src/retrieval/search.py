from collections.abc import Mapping

from models import SearchResult


def filter_results(
    results: list[SearchResult], metadata: Mapping[str, str] | None = None
) -> list[SearchResult]:
    if not metadata:
        return results
    return [
        result for result in results
        if all(getattr(result.chunk.metadata, key, None) == value
               for key, value in metadata.items())
    ]
