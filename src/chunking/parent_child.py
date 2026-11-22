"""Fast, table-aware Parent-Child chunker calibrated for BAAI/bge-m3."""

import logging
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import get_settings
from domain import ChildChunk, ParentChunk, ParsedDoc, ParsedSection

logger = logging.getLogger(__name__)

_MD_TABLE_SEP_RE = re.compile(r"^\|\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?$")


def approx_token_len(text: str) -> int:
    """Fast pure-Python token-length estimate calibrated for BGE-M3 prose and Markdown tables."""
    if not text:
        return 0
    return max(1, (len(text) + 2) // 3 + text.count("|"))


@lru_cache(maxsize=2)
def load_tokenizer(model_name: str = "BAAI/bge-m3") -> PreTrainedTokenizerBase:
    """Lazy-load and cache the HuggingFace tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 1_000_000
    return tokenizer


def _is_markdown_table(text: str) -> bool:
    """Return True if `text` starts with a Markdown table header + separator row."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return lines[0].startswith("|") and bool(_MD_TABLE_SEP_RE.match(lines[1]))


class ParentChildChunker:
    """Splits a `ParsedDoc` into context-rich `ParentChunk`s and overlapping `ChildChunk`s."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        parent_size: int = 1000,
        parent_overlap: int = 0,
        child_size: int = 250,
        child_overlap: int = 40,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_child_chunks_per_doc: int | None = None,
    ) -> None:
        if parent_size <= 0 or child_size <= 0 or child_size > parent_size:
            raise ValueError("Chunk sizes must be positive and child_size <= parent_size.")
        if parent_overlap >= parent_size or child_overlap >= child_size:
            raise ValueError("Overlap must be strictly smaller than the respective chunk size.")

        self.model_name = model_name
        self._tokenizer: PreTrainedTokenizerBase | None = tokenizer
        self.parent_size = parent_size
        self.parent_overlap = parent_overlap
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.max_child_chunks_per_doc = (
            max_child_chunks_per_doc
            if max_child_chunks_per_doc is not None
            else get_settings().max_child_chunks_per_doc
        )

        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            length_function=approx_token_len,
            keep_separator=True,
            separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            length_function=approx_token_len,
            keep_separator=False,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Lazily load and cache the HuggingFace tokenizer when exact token counts are requested."""
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.model_name)
        return self._tokenizer

    def _token_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _is_short_heading_or_intro(self, text: str, token_len: int) -> bool:
        """Return True if a prose section is a detached short heading or lead-in line."""
        if token_len > min(40, max(10, self.parent_size // 3)):
            return False
        return text.startswith("#") or text.endswith(":") or ("\n" not in text)

    def _pack_sections(self, raw_sections: Sequence[ParsedSection]) -> list[ParsedSection]:
        """Attach orphan headings/intros to following content and pack consecutive short prose."""
        cleaned: list[ParsedSection] = []
        for sec in raw_sections:
            text = sec.text.replace("\r\n", "\n").strip()
            if not text:
                continue
            cleaned.append(
                ParsedSection(
                    text=text,
                    page_numbers=list(sec.page_numbers),
                    section_header=sec.section_header,
                    block_type=sec.block_type,
                    sheet_name=sec.sheet_name,
                )
            )
        if len(cleaned) <= 1:
            return cleaned

        # Pass 1: Attach detached short headings/intros to the immediately following prose or table.
        attached: list[ParsedSection] = []
        i = 0
        while i < len(cleaned):
            curr = cleaned[i]
            curr_is_prose = curr.block_type == "text" and not _is_markdown_table(curr.text)
            if (
                curr_is_prose
                and i + 1 < len(cleaned)
                and self._is_short_heading_or_intro(curr.text, approx_token_len(curr.text))
            ):
                nxt = cleaned[i + 1]
                nxt_is_table = nxt.block_type == "table" or _is_markdown_table(nxt.text)
                merged_pages = sorted(set(curr.page_numbers) | set(nxt.page_numbers))
                if nxt_is_table:
                    intro_label = re.sub(r"^#+\s*", "", curr.text).strip().rstrip(":")
                    if not nxt.section_header:
                        header = intro_label or curr.section_header
                    elif intro_label and intro_label.lower() not in nxt.section_header.lower():
                        header = f"{nxt.section_header} — {intro_label}"
                    else:
                        header = nxt.section_header
                    cleaned[i + 1] = ParsedSection(
                        text=nxt.text,
                        page_numbers=merged_pages,
                        section_header=header,
                        block_type="table",
                        sheet_name=nxt.sheet_name or curr.sheet_name,
                    )
                    i += 1
                    continue
                if nxt.block_type == "text":
                    cleaned[i + 1] = ParsedSection(
                        text=f"{curr.text}\n\n{nxt.text}",
                        page_numbers=merged_pages,
                        section_header=curr.section_header or nxt.section_header,
                        block_type="text",
                        sheet_name=curr.sheet_name or nxt.sheet_name,
                    )
                    i += 1
                    continue
            attached.append(curr)
            i += 1

        # Pass 2: Pack consecutive short prose sections while staying within parent_size.
        min_pack_tokens = min(150, self.parent_size // 2)
        packed: list[ParsedSection] = []
        last_tokens = 0
        for sec in attached:
            is_prose = sec.block_type == "text" and not _is_markdown_table(sec.text)
            sec_tokens = approx_token_len(sec.text) if is_prose else self.parent_size + 1
            if packed and is_prose:
                prev = packed[-1]
                prev_is_prose = prev.block_type == "text" and not _is_markdown_table(prev.text)
                if (
                    prev_is_prose
                    and (last_tokens < min_pack_tokens or sec_tokens < min_pack_tokens)
                    and (last_tokens + sec_tokens + 2) <= self.parent_size
                ):
                    merged_text = f"{prev.text}\n\n{sec.text}"
                    packed[-1] = ParsedSection(
                        text=merged_text,
                        page_numbers=sorted(set(prev.page_numbers) | set(sec.page_numbers)),
                        section_header=prev.section_header or sec.section_header,
                        block_type="text",
                        sheet_name=prev.sheet_name or sec.sheet_name,
                    )
                    last_tokens = approx_token_len(merged_text)
                    continue
            packed.append(sec)
            last_tokens = sec_tokens

        return packed

    def _split_markdown_table(
        self,
        table_text: str,
        max_tokens: int,
        overlap_rows: int = 0,
    ) -> list[str]:
        """Split a Markdown table by row boundaries, repeating the header unless it is too wide."""
        lines = [line.strip() for line in table_text.splitlines() if line.strip()]
        if len(lines) <= 2 or approx_token_len(table_text) <= max_tokens:
            return [table_text]

        header_block = f"{lines[0]}\n{lines[1]}"
        body_rows = lines[2:]
        header_tokens = approx_token_len(header_block)

        # Guard against wide Excel tables (e.g. 25+ columns) whose header consumes >40% of
        # max_tokens, which would otherwise force every row into header-dominated micro-chunks.
        repeat_header = header_tokens <= max(45, int(max_tokens * 0.4))
        effective_header = header_block if repeat_header else ""
        effective_header_tokens = header_tokens if repeat_header else 0

        chunks: list[str] = []
        idx = 0
        total_rows = len(body_rows)

        while idx < total_rows:
            current_rows: list[str] = []
            current_tokens = effective_header_tokens
            start_idx = idx

            while idx < total_rows:
                row = body_rows[idx]
                row_tokens = approx_token_len(row) + 1
                if current_rows and (current_tokens + row_tokens) > max_tokens:
                    break
                current_rows.append(row)
                current_tokens += row_tokens
                idx += 1

            if len(current_rows) == 1 and current_tokens > max_tokens:
                row_budget = max(20, max_tokens - effective_header_tokens - 2)
                row_overlap = min(self.child_overlap, row_budget // 4) if overlap_rows > 0 else 0
                row_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=row_budget,
                    chunk_overlap=row_overlap,
                    length_function=approx_token_len,
                    keep_separator=False,
                    separators=[" | ", "; ", ". ", " ", ""],
                )
                for sub_row in row_splitter.split_text(current_rows[0]):
                    sub_clean = sub_row.strip()
                    if sub_clean:
                        chunks.append(
                            f"{effective_header}\n{sub_clean}" if effective_header else sub_clean
                        )
            else:
                body_block = "\n".join(current_rows)
                chunks.append(
                    f"{effective_header}\n{body_block}" if effective_header else body_block
                )
                if idx < total_rows and overlap_rows > 0 and len(current_rows) > overlap_rows:
                    idx = max(start_idx + 1, idx - overlap_rows)

        return chunks

    def chunk(self, document: ParsedDoc) -> tuple[list[ParentChunk], list[ChildChunk]]:
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        raw_sections = document.sections or [ParsedSection(text=document.markdown)]
        sections = self._pack_sections(raw_sections)
        parent_idx = 0
        cap = self.max_child_chunks_per_doc
        capped = False

        for section in sections:
            if capped:
                break
            raw_text = section.text.replace("\r\n", "\n").strip()
            if not raw_text:
                continue

            is_table = (
                section.block_type == "table" and _is_markdown_table(raw_text)
            ) or _is_markdown_table(raw_text)
            if is_table:
                parent_texts = self._split_markdown_table(
                    raw_text, max_tokens=self.parent_size, overlap_rows=0
                )
            else:
                clean_text = f"\n{raw_text}" if raw_text.startswith("#") else raw_text
                parent_texts = self._parent_splitter.split_text(clean_text)

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

                new_children: list[ChildChunk] = []
                current_child_ids: list[str] = []
                for child_idx, c_text in enumerate(child_texts):
                    c_text = c_text.strip()
                    if not c_text:
                        continue

                    child_id = f"{parent_id}:c{child_idx}"
                    current_child_ids.append(child_id)
                    new_children.append(
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

                if cap > 0 and parents and (len(children) + len(new_children)) > cap:
                    logger.warning(
                        "Document '%s' (%s) reached max_child_chunks_per_doc=%d "
                        "(kept %d parents / %d children); truncating remaining sections.",
                        document.metadata.source_path,
                        document.document_id,
                        cap,
                        len(parents),
                        len(children),
                    )
                    capped = True
                    break

                children.extend(new_children)
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
    def _build_embedding_text(text: str, document: ParsedDoc, section_header: str | None) -> str:
        """Prefix child text with normalized metadata tags for dense + sparse BGE-M3 encoding."""
        meta = document.metadata
        file_stem = Path(meta.source_path).stem if meta.source_path else None
        include_building = bool(meta.building) and len(meta.buildings) <= 3
        tags = [
            f"File: {file_stem}" if file_stem else None,
            f"Project: {meta.project}" if meta.project else None,
            f"Building: {meta.building}" if include_building else None,
            f"Phase: {meta.phase}" if meta.phase else None,
            f"Section: {section_header}" if section_header else None,
        ]
        active = [t for t in tags if t is not None]
        return f"[{' | '.join(active)}]\n{text}" if active else text
