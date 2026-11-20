"""Token-accurate, table-aware Parent-Child chunker aligned with BAAI/bge-m3."""

from functools import lru_cache
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from domain import ChildChunk, ParentChunk, ParsedDoc, ParsedSection

_MD_TABLE_SEP_RE = re.compile(r"^\|\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?$")


@lru_cache(maxsize=2)
def load_tokenizer(model_name: str = "BAAI/bge-m3") -> PreTrainedTokenizerBase:
    """Lazy-loads and caches the HuggingFace tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 1_000_000
    return tokenizer


def _is_markdown_table(text: str) -> bool:
    """Return True if `text` is a Markdown table block (header row + separator row)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return lines[0].startswith("|") and bool(_MD_TABLE_SEP_RE.match(lines[1]))


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

    def _token_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_markdown_table(
        self,
        table_text: str,
        max_tokens: int,
        overlap_rows: int = 0,
    ) -> list[str]:
        """Split a Markdown table by row boundaries while repeating the 2-line header in every chunk."""
        lines = [line.strip() for line in table_text.splitlines() if line.strip()]
        if len(lines) <= 2:
            return [table_text]

        header_block = f"{lines[0]}\n{lines[1]}"
        body_rows = lines[2:]
        if self._token_len(table_text) <= max_tokens:
            return [table_text]

        header_tokens = self._token_len(header_block)
        chunks: list[str] = []
        idx = 0
        total_rows = len(body_rows)

        while idx < total_rows:
            current_rows: list[str] = []
            current_tokens = header_tokens
            start_idx = idx

            while idx < total_rows:
                row = body_rows[idx]
                row_tokens = self._token_len(row) + 1  # +1 for newline
                if current_rows and (current_tokens + row_tokens) > max_tokens:
                    break
                current_rows.append(row)
                current_tokens += row_tokens
                idx += 1

            chunks.append(header_block + "\n" + "\n".join(current_rows))

            if idx < total_rows and overlap_rows > 0 and len(current_rows) > overlap_rows:
                idx = max(start_idx + 1, idx - overlap_rows)

        return chunks

    def chunk(self, document: ParsedDoc) -> tuple[list[ParentChunk], list[ChildChunk]]:
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        sections = document.sections or [ParsedSection(text=document.markdown)]
        parent_idx = 0

        for section in sections:
            raw_text = section.text.replace("\r\n", "\n").strip()
            if not raw_text:
                continue

            is_table = _is_markdown_table(raw_text)
            if is_table:
                parent_texts = self._split_markdown_table(
                    raw_text, max_tokens=self.parent_size, overlap_rows=0
                )
            else:
                # Prepend newline if starting with '#' so '\n# ' separators match line 1
                clean_section_text = f"\n{raw_text}" if raw_text.startswith("#") else raw_text
                parent_texts = self._parent_splitter.split_text(clean_section_text)

            for p_text in parent_texts:
                p_text = p_text.strip()
                if not p_text:
                    continue

                parent_id = f"{document.document_id}:p{parent_idx}"
                if is_table:
                    child_texts = self._split_markdown_table(
                        p_text,
                        max_tokens=self.child_size,
                        overlap_rows=1 if self.child_overlap > 0 else 0,
                    )
                else:
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
                            metadata=document.metadata.model_copy(deep=True),
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
                        metadata=document.metadata.model_copy(deep=True),
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
