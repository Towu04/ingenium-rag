from ingenium_rag.models import Answer, SearchResult


class GroundedGenerator:
    def generate(self, question: str, results: list[SearchResult]) -> Answer:
        if not results:
            return Answer(text="I could not find supporting information in the indexed documents.")
        citations = [
            {"document_id": result.chunk.document_id,
             "source_path": result.chunk.metadata.source_path,
             "page_number": result.chunk.metadata.page_number,
             "chunk_id": result.chunk.id}
            for result in results
        ]
        context = "\n\n".join(result.chunk.text for result in results)
        return Answer(text=f"Context found for '{question}':\n{context}", citations=citations)
