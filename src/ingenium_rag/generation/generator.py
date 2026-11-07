from ingenium_rag.domain import Answer, SearchResult
from ingenium_rag.generation.citations import citations_for


class GroundedGenerator:
    def generate(self, question: str, results: list[SearchResult]) -> Answer:
        if not results:
            return Answer(text="I could not find supporting information in the indexed documents.")
        context = "\n\n".join(result.chunk.text for result in results)
        return Answer(
            text=f"Context found for '{question}':\n{context}",
            citations=citations_for(results),
        )
