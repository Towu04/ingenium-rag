"""Offline indexing pipeline: ETL -> Adapter -> ParentChildChunker -> BGEM3Embedder -> LocalHybridRepository."""

from collections.abc import Iterable
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Protocol, Sequence

from domain import ChildChunk, ParsedDoc
from processing.chunker import ParentChildChunker
from processing.etl_adapter import (
    adapt_extracted_document,
    load_corpus_from_etl_dir,
)
from storage.embedder import BGEM3Embedder, EmbeddedChildBatch
from storage.repository import LocalHybridRepository

logger = logging.getLogger(__name__)


class ChildEmbedder(Protocol):
    """Minimal embedder protocol satisfied by `BGEM3Embedder` and lightweight test fakes."""

    def embed_children(self, children: Sequence[ChildChunk]) -> EmbeddedChildBatch: ...


@dataclass
class IndexingStats:
    """Execution summary returned by `IndexingPipeline`."""

    total_docs: int = 0
    indexed_docs: int = 0
    skipped_unchanged: int = 0
    skipped_empty: int = 0
    failed_docs: int = 0
    total_parents: int = 0
    total_children: int = 0


class IndexingPipeline:
    """Orchestrates incremental Parent-Child chunking, hybrid embedding, and Qdrant persistence."""

    def __init__(
        self,
        repository: LocalHybridRepository,
        embedder: ChildEmbedder | None = None,
        chunker: ParentChildChunker | None = None,
    ) -> None:
        self.repository = repository
        self.embedder: ChildEmbedder = embedder or BGEM3Embedder()
        if chunker is not None:
            self.chunker = chunker
        else:
            shared_tokenizer = getattr(self.embedder, "tokenizer", None)
            self.chunker = ParentChildChunker(tokenizer=shared_tokenizer)

    def index_documents(
        self,
        documents: Iterable[ParsedDoc],
        *,
        force: bool = False,
    ) -> IndexingStats:
        """Index an iterable of `ParsedDoc`s into Qdrant, respecting `needs_indexing`."""
        stats = IndexingStats()

        for doc in documents:
            stats.total_docs += 1
            try:
                if not force and not self.repository.needs_indexing(
                    doc.document_id, doc.content_hash
                ):
                    stats.skipped_unchanged += 1
                    logger.debug(
                        "Skipping unchanged document '%s' (%s).",
                        doc.metadata.source_path,
                        doc.document_id,
                    )
                    continue

                parents, children = self.chunker.chunk(doc)
                if not parents or not children:
                    stats.skipped_empty += 1
                    logger.warning(
                        "Skipping document with 0 chunks after chunking: '%s' (%s).",
                        doc.metadata.source_path,
                        doc.document_id,
                    )
                    continue

                embedded_batch = self.embedder.embed_children(children)
                self.repository.upsert_document_bundle(doc, parents, children, embedded_batch)

                stats.indexed_docs += 1
                stats.total_parents += len(parents)
                stats.total_children += len(children)
                logger.info(
                    "Indexed '%s' (%s): %d parents, %d children.",
                    doc.metadata.source_path,
                    doc.document_id,
                    len(parents),
                    len(children),
                )
            except Exception:
                stats.failed_docs += 1
                logger.exception(
                    "Failed to index document '%s' (%s).",
                    doc.metadata.source_path,
                    doc.document_id,
                )

        return stats

    def ingest_from_etl_dir(
        self,
        processed_dir: Path | str,
        *,
        force: bool = False,
        use_manifest: bool = True,
        dedupe_by_hash: bool = True,
        drop_noisy_callouts: bool = True,
    ) -> IndexingStats:
        """Load pre-extracted ETL JSONs from `processed_dir` and index them into Qdrant."""
        docs = load_corpus_from_etl_dir(
            processed_dir,
            use_manifest=use_manifest,
            dedupe_by_hash=dedupe_by_hash,
            drop_noisy_callouts=drop_noisy_callouts,
        )
        return self.index_documents(docs, force=force)

    def ingest_raw_folder(
        self,
        raw_dir: Path | str,
        *,
        catalog_path: Path | str | None = None,
        processed_dir: Path | str = "data/processed",
        force_etl: bool = False,
        force_index: bool = False,
        drop_noisy_callouts: bool = True,
    ) -> IndexingStats:
        """Run `ETLPipeline` on a raw folder, adapt newly extracted docs, and index them."""
        from etl.pipeline import ETLPipeline

        out_base = Path(processed_dir)
        etl = ETLPipeline(
            catalog_path=catalog_path,
            markdown_out_dir=out_base / "markdown",
            json_out_dir=out_base / "json",
            manifest_path=out_base / "manifest.json",
        )
        run_result = etl.run(raw_dir, force=force_etl)
        if run_result.extracted_documents:
            docs = [
                adapt_extracted_document(d, drop_noisy_callouts=drop_noisy_callouts)
                for d in run_result.extracted_documents
                if d.blocks
            ]
            return self.index_documents(docs, force=force_index)

        # If ETL skipped everything as unchanged, still ensure Qdrant is synced with manifest
        return self.ingest_from_etl_dir(
            out_base,
            force=force_index,
            drop_noisy_callouts=drop_noisy_callouts,
        )
