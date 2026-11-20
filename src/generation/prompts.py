"""Prompt templates for citation-grounded engineering QA."""

from collections.abc import Sequence

from domain import SearchResult

SYSTEM_PROMPT = """You are an engineering assistant for Ingenium (HVAC, electrical, sanitary, and fire safety projects).
Answer strictly from the numbered context passages supplied below.
- Respond in the same language as the user's question (Dutch or English).
- Every factual claim, specification, or number MUST include an inline citation such as [1] or [2].
- If the supplied context does not contain enough information to answer, say so explicitly.
- Never invent values, equipment IDs, or sources."""


def build_user_prompt(question: str, results: Sequence[SearchResult]) -> str:
    """Format numbered parent-chunk context blocks with provenance headers for LLM generation."""
    blocks: list[str] = []
    for idx, result in enumerate(results, start=1):
        parent = result.parent_chunk
        meta = parent.metadata
        pages = (
            f"Pages {', '.join(map(str, parent.page_numbers))}"
            if parent.page_numbers
            else "Pages N/A"
        )
        section = parent.section_header or "General"
        header_parts = [
            f"[{idx}] Source: {meta.source_path}",
            pages,
            f"Section: {section}",
        ]
        if meta.project:
            header_parts.append(f"Project: {meta.project}")
        if meta.building:
            header_parts.append(f"Building: {meta.building}")
        if meta.phase:
            header_parts.append(f"Phase: {meta.phase}")

        blocks.append(f"{' | '.join(header_parts)}\n{parent.text}")

    context_str = "\n\n---\n\n".join(blocks)
    return f"Context:\n{context_str}\n\nQuestion:\n{question}"
