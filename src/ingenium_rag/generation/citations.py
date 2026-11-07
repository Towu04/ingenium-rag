from ingenium_rag.domain import Citation, SearchResult


def citations_for(results: list[SearchResult]) -> list[Citation]:
    return [
        Citation(
            document_id=result.chunk.document_id,
            source_path=result.chunk.metadata.source_path,
            page_number=result.chunk.metadata.page_number,
            section=result.chunk.metadata.section,
            chunk_id=result.chunk.id,
        )
        for result in results
    ]


def format_citation(citation: Citation) -> str:
    page = f", Page {citation.page_number}" if citation.page_number is not None else ""
    section = f", {citation.section}" if citation.section else ""
    return f"[{citation.source_path}{page}{section}]"
