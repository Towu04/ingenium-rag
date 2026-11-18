"""Token-accurate Parent-Child chunker aligned with BAAI/bge-m3."""

from functools import lru_cache
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from domain import ChildChunk, ParentChunk, ParsedDoc, ParsedSection


@lru_cache(maxsize=2)
def load_tokenizer(model_name: str = "BAAI/bge-m3") -> PreTrainedTokenizerBase:
    """Lazy-loads and caches the HuggingFace tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 1_000_000
    return tokenizer


class ParentChildChunker:
    """Splits a ParsedDoc into context-rich ParentChunks and overlapping ChildChunks."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        parent_size: int = 1000,
        parent_overlap: int = 0,
        child_size: int = 250,
        child_overlap: int = 40,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        if parent_size <= 0 or child_size <= 0 or child_size > parent_size:
            raise ValueError("Chunk sizes must be positive and child_size <= parent_size.")
        if parent_overlap >= parent_size or child_overlap >= child_size:
            raise ValueError("Overlap must be strictly smaller than the respective chunk size.")

        self.tokenizer = tokenizer or load_tokenizer(model_name)
        self.parent_size = parent_size
        self.parent_overlap = parent_overlap
        self.child_size = child_size
        self.child_overlap = child_overlap

        self._parent_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            keep_separator=True,
            separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )

        self._child_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            keep_separator=False,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, document: ParsedDoc) -> tuple[list[ParentChunk], list[ChildChunk]]:
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        sections = document.sections or [ParsedSection(text=document.markdown)]
        doc_metadata = document.metadata.model_copy(deep=True)
        parent_idx = 0

        for section in sections:
            raw_text = section.text.replace("\r\n", "\n").strip()
            if not raw_text:
                continue

            # Prepend newline if starting with '#' so '\n# ' separators match line 1
            clean_section_text = f"\n{raw_text}" if raw_text.startswith("#") else raw_text
            parent_texts = self._parent_splitter.split_text(clean_section_text)

            for p_text in parent_texts:
                p_text = p_text.strip()
                if not p_text:
                    continue

                parent_id = f"{document.document_id}:p{parent_idx}"
                child_texts = self._child_splitter.split_text(p_text)
                current_child_ids: list[str] = []

                for child_idx, c_text in enumerate(child_texts):
                    c_text = c_text.strip()
                    if not c_text:
                        continue

                    child_id = f"{parent_id}:c{child_idx}"
                    current_child_ids.append(child_id)

                    children.append(
                        ChildChunk(
                            id=child_id,
                            parent_id=parent_id,
                            document_id=document.document_id,
                            child_index=child_idx,
                            text=c_text,
                            embedding_text=self._build_embedding_text(
                                c_text, document, section.section_header
                            ),
                            page_numbers=list(section.page_numbers),
                            section_header=section.section_header,
                            metadata=doc_metadata,
                        )
                    )

                if not current_child_ids:
                    continue

                parents.append(
                    ParentChunk(
                        id=parent_id,
                        document_id=document.document_id,
                        parent_index=parent_idx,
                        text=p_text,
                        page_numbers=list(section.page_numbers),
                        section_header=section.section_header,
                        child_ids=current_child_ids,
                        metadata=doc_metadata,
                    )
                )
                parent_idx += 1

        return parents, children

    @staticmethod
    def _build_embedding_text(
        text: str, document: ParsedDoc, section_header: str | None
    ) -> str:
        """Prepends metadata context for BGE-M3 dense + sparse encoding."""
        meta = document.metadata
        tags = [
            f"Project: {meta.project}" if meta.project else None,
            f"Building: {meta.building}" if meta.building else None,
            f"Phase: {meta.phase}" if meta.phase else None,
            f"Section: {section_header}" if section_header else None,
        ]
        active_tags = [t for t in tags if t is not None]
        if not active_tags:
            return text
        return f"[{' | '.join(active_tags)}]\n{text}"
