from domain import Citation, SearchResult


def citations_for(results: list[SearchResult]) -> list[Citation]:
    return [
        Citation(
            document_id=result.parent_chunk.document_id,
            source_path=result.parent_chunk.metadata.source_path,
            sharepoint_url=result.parent_chunk.metadata.sharepoint_url,
            page_numbers=result.parent_chunk.page_numbers,
            section_header=result.parent_chunk.section_header,
            parent_chunk_id=result.parent_chunk.id,
        )
        for result in results
    ]


def format_citation(citation: Citation) -> str:
    pages = f", Pages {', '.join(map(str, citation.page_numbers))}" if citation.page_numbers else ""
    section = f", {citation.section_header}" if citation.section_header else ""
    return f"[{citation.source_path}{pages}{section}]"
