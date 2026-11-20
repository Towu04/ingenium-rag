"""Thin top-level orchestrator wiring isolated stages via canonical `domain` contracts.

This is the ONLY library module that imports across stages (`etl`, `chunking`,
`embedding`, `storage`, `retrieval`, `generation`). Every stage remains strictly
isolated and testable in complete independence.
"""

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from chunking import ParentChildChunker
from config import Settings, get_settings
from domain import (
    Answer,
    ChildChunk,
    EmbeddedChildBatch,
    ParentChunk,
    ParsedDoc,
    RetrievalTrace,
    SearchResult,
)
from embedding import BGEM3Embedder
from etl import ETLRunner, SharePointConnector, load_processed_docs
from generation.generator import (
    ExtractiveStubLLM,
    GroundedGenerator,
    LLMClient,
    OpenAICompatibleLLM,
)
from retrieval import BGEM3Reranker, Searcher, extract_metadata_filters
from storage import LocalHybridRepository

logger = logging.getLogger(__name__)


class ChildEmbedder(Protocol):
    def embed_children(self, children: Sequence[ChildChunk]) -> EmbeddedChildBatch: ...


class RerankerProtocol(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[SearchResult], top_k: int = 5
    ) -> list[SearchResult]: ...


@dataclass
class IndexingStats:
    """Execution summary for an `IndexingPipeline` run."""

    total_docs: int = 0
    indexed_docs: int = 0
    skipped_unchanged: int = 0
    skipped_empty: int = 0
    failed_docs: int = 0
    total_parents: int = 0
    total_children: int = 0


class IndexingPipeline:
    """Offline pipeline: `ParsedDoc` -> `ParentChildChunker` -> `BGEM3Embedder` -> `LocalHybridRepository`."""

    def __init__(
        self,
        repository: LocalHybridRepository,
        embedder: ChildEmbedder | None = None,
        chunker: ParentChildChunker | None = None,
        *,
        device: str | None = None,
        embed_batch_buffer_size: int | None = None,
    ) -> None:
        self._settings = get_settings()
        self.repository = repository
        self._device = device
        self._embedder: ChildEmbedder | None = embedder
        self._chunker: ParentChildChunker | None = chunker
        self.embed_batch_buffer_size = (
            embed_batch_buffer_size
            if embed_batch_buffer_size is not None
            else self._settings.embed_batch_buffer_size
        )

    @property
    def embedder(self) -> ChildEmbedder:
        """Lazily load `BGEM3Embedder` only when at least one chunk actually needs embedding."""
        if self._embedder is None:
            self._embedder = BGEM3Embedder(
                model_name=self._settings.embedding_model,
                device=self._device or self._settings.device,
                batch_size=self._settings.embedding_batch_size,
                max_length=self._settings.embedding_max_length,
            )
        return self._embedder

    @property
    def chunker(self) -> ParentChildChunker:
        """Lazily initialize `ParentChildChunker` without forcing embedder weight loading."""
        if self._chunker is None:
            self._chunker = ParentChildChunker(
                model_name=self._settings.embedding_model,
                parent_size=self._settings.parent_size,
                parent_overlap=self._settings.parent_overlap,
                child_size=self._settings.child_size,
                child_overlap=self._settings.child_overlap,
                tokenizer=getattr(self._embedder, "tokenizer", None),
            )
        return self._chunker

    def index_documents(
        self,
        documents: Iterable[ParsedDoc],
        *,
        force: bool = False,
    ) -> IndexingStats:
        """Index `ParsedDoc`s into Qdrant using length-sorted cross-document batch embedding."""
        t_start = time.perf_counter()
        stats = IndexingStats()
        buffer: list[tuple[ParsedDoc, list[ParentChunk], list[ChildChunk]]] = []
        buffered_children = 0

        def flush_buffer() -> None:
            nonlocal buffered_children
            if not buffer:
                return

            flat_children: list[ChildChunk] = []
            slices: list[tuple[int, int]] = []
            for _, _, doc_children in buffer:
                start = len(flat_children)
                flat_children.extend(doc_children)
                slices.append((start, len(flat_children)))

            try:
                t_embed = time.perf_counter()
                order = sorted(
                    range(len(flat_children)),
                    key=lambda i: len(flat_children[i].embedding_text or flat_children[i].text),
                )
                sorted_children = [flat_children[i] for i in order]
                sorted_batch = self.embedder.embed_children(sorted_children)

                dense_out = np.empty_like(sorted_batch.dense_vecs)
                sparse_out: list[dict[int, float]] = [{} for _ in range(len(flat_children))]
                for sorted_pos, orig_pos in enumerate(order):
                    dense_out[orig_pos] = sorted_batch.dense_vecs[sorted_pos]
                    sparse_out[orig_pos] = sorted_batch.sparse_weights[sorted_pos]
                logger.info(
                    "Embedded %d child chunk(s) across %d doc(s) in %.1fs.",
                    len(flat_children),
                    len(buffer),
                    time.perf_counter() - t_embed,
                )
            except Exception:
                stats.failed_docs += len(buffer)
                logger.exception("Batch embedding failed for %d buffered document(s).", len(buffer))
                buffer.clear()
                buffered_children = 0
                return

            for (doc, parents, doc_children), (start, end) in zip(buffer, slices):
                try:
                    doc_batch = EmbeddedChildBatch(
                        dense_vecs=dense_out[start:end],
                        sparse_weights=sparse_out[start:end],
                    )
                    self.repository.upsert_document_bundle(doc, parents, doc_children, doc_batch)
                    stats.indexed_docs += 1
                    stats.total_parents += len(parents)
                    stats.total_children += len(doc_children)
                    logger.debug(
                        "Indexed '%s' (%s): %d parents, %d children.",
                        doc.metadata.source_path,
                        doc.document_id,
                        len(parents),
                        len(doc_children),
                    )
                except Exception:
                    stats.failed_docs += 1
                    logger.exception(
                        "Failed to upsert '%s' (%s).", doc.metadata.source_path, doc.document_id
                    )

            buffer.clear()
            buffered_children = 0

        for doc in documents:
            stats.total_docs += 1
            try:
                if not force and not self.repository.needs_indexing(
                    doc.document_id, doc.content_hash
                ):
                    stats.skipped_unchanged += 1
                    logger.debug(
                        "Skipping unchanged '%s' (%s).", doc.metadata.source_path, doc.document_id
                    )
                    continue

                parents, children = self.chunker.chunk(doc)
                if not parents or not children:
                    stats.skipped_empty += 1
                    logger.warning(
                        "Skipping '%s' (%s): 0 chunks generated.",
                        doc.metadata.source_path,
                        doc.document_id,
                    )
                    continue

                buffer.append((doc, parents, children))
                buffered_children += len(children)
                if buffered_children >= self.embed_batch_buffer_size:
                    flush_buffer()
            except Exception:
                stats.failed_docs += 1
                logger.exception(
                    "Failed to prepare '%s' (%s).", doc.metadata.source_path, doc.document_id
                )

        flush_buffer()
        logger.info(
            "Indexing complete in %.1fs: %d indexed, %d unchanged, %d empty, %d failed "
            "(%d parents, %d children).",
            time.perf_counter() - t_start,
            stats.indexed_docs,
            stats.skipped_unchanged,
            stats.skipped_empty,
            stats.failed_docs,
            stats.total_parents,
            stats.total_children,
        )
        return stats

    def ingest_processed_dir(
        self,
        processed_dir: str | Path,
        *,
        force: bool = False,
        use_manifest: bool = True,
        dedupe_by_hash: bool = True,
    ) -> IndexingStats:
        """Load persisted `ParsedDoc` JSONs from `processed_dir` and index them."""
        docs = load_processed_docs(
            processed_dir,
            use_manifest=use_manifest,
            dedupe_by_hash=dedupe_by_hash,
        )
        return self.index_documents(docs, force=force)

    def ingest_raw_folder(
        self,
        raw_dir: str | Path,
        *,
        catalog_path: str | Path | None = None,
        processed_dir: str | Path | None = None,
        force_etl: bool = False,
        force_index: bool = False,
        drop_noisy_callouts: bool = True,
    ) -> IndexingStats:
        """Run local ETL extraction into `processed_dir`, then index all valid `ParsedDoc`s."""
        out_base = (
            Path(processed_dir) if processed_dir is not None else get_settings().processed_dir
        )
        etl = ETLRunner(
            catalog_path=catalog_path,
            markdown_out_dir=out_base / "markdown",
            json_out_dir=out_base / "json",
            manifest_path=out_base / "manifest.json",
            drop_noisy_callouts=drop_noisy_callouts,
        )
        etl.run_local(raw_dir, force=force_etl)
        return self.ingest_processed_dir(out_base, force=force_index)

    def ingest_sharepoint(
        self,
        *,
        catalog_path: str | Path | None = None,
        processed_dir: str | Path | None = None,
        force_etl: bool = False,
        force_index: bool = False,
        max_files: int | None = None,
        folder_prefix: str | None = None,
        drop_noisy_callouts: bool = True,
    ) -> IndexingStats:
        """Stream files from SharePoint via ETL, persist `ParsedDoc` JSONs, and index into Qdrant."""
        out_base = (
            Path(processed_dir) if processed_dir is not None else get_settings().processed_dir
        )
        etl = ETLRunner(
            catalog_path=catalog_path,
            markdown_out_dir=out_base / "markdown",
            json_out_dir=out_base / "json",
            manifest_path=out_base / "manifest.json",
            drop_noisy_callouts=drop_noisy_callouts,
        )
        connector = SharePointConnector.from_env(json_out_dir=out_base / "json")
        etl.run_sharepoint(
            connector=connector,
            force=force_etl,
            max_files=max_files,
            folder_prefix=folder_prefix,
        )
        return self.ingest_processed_dir(out_base, force=force_index)


class RAGPipeline:
    """Online pipeline: Query -> Filter -> Hybrid Search -> Cross-Encoder Rerank -> Grounded Answer."""

    def __init__(
        self,
        searcher: Searcher,
        reranker: RerankerProtocol | None = None,
        generator: GroundedGenerator | None = None,
        retrieve_k: int = 15,
        child_candidate_limit: int = 80,
        rerank_k: int = 5,
    ) -> None:
        settings = get_settings()
        self.searcher = searcher
        self.reranker: RerankerProtocol = reranker or BGEM3Reranker(
            model_name=settings.reranker_model,
            device=settings.device,
            batch_size=settings.reranker_batch_size,
            max_length=settings.reranker_max_length,
            min_score=settings.reranker_min_score,
        )
        self.generator = generator or GroundedGenerator()
        self.retrieve_k = retrieve_k
        self.child_candidate_limit = child_candidate_limit
        self.rerank_k = rerank_k

    def retrieve_with_trace(
        self,
        query: str,
        *,
        top_k: int | None = None,
        retrieve_k: int | None = None,
        project: str | None = None,
        building: str | None = None,
        phase: str | None = None,
        auto_filter: bool = False,
        min_score: float | None = None,
    ) -> RetrievalTrace:
        """Run hybrid search + reranking and return a full `RetrievalTrace` for inspection/debugging."""
        explicit = {
            k: v
            for k, v in (("project", project), ("building", building), ("phase", phase))
            if v is not None
        }
        known_buildings_fn = getattr(self.searcher.repository, "known_buildings", None)
        known_bldgs = (
            known_buildings_fn() if (auto_filter and callable(known_buildings_fn)) else None
        )
        inferred = (
            extract_metadata_filters(query, known_buildings=known_bldgs) if auto_filter else {}
        )
        eff_project = project if project is not None else inferred.get("project")
        eff_building = building if building is not None else inferred.get("building")
        eff_phase = phase if phase is not None else inferred.get("phase")
        applied = {
            k: v
            for k, v in (
                ("project", eff_project),
                ("building", eff_building),
                ("phase", eff_phase),
            )
            if v is not None
        }
        used_only_inferred = auto_filter and not explicit

        k_final = top_k if top_k is not None else self.rerank_k
        k_retrieve = max(retrieve_k if retrieve_k is not None else self.retrieve_k, k_final)

        candidates, child_hits, fell_back = self.searcher.search_with_stats(
            query,
            top_k_parents=k_retrieve,
            child_candidate_limit=self.child_candidate_limit,
            project=eff_project,
            building=eff_building,
            phase=eff_phase,
            fallback_unfiltered=used_only_inferred,
        )

        eff_min_score = (
            min_score if min_score is not None else float(getattr(self.reranker, "min_score", 0.0))
        )
        if not candidates:
            all_kept: list[SearchResult] = []
        elif min_score is not None:
            all_kept = self.reranker.rerank(
                query, candidates, top_k=len(candidates), min_score=min_score
            )
        else:
            all_kept = self.reranker.rerank(query, candidates, top_k=len(candidates))

        final_results = all_kept[:k_final]
        return RetrievalTrace(
            query=query,
            explicit_filters=explicit,
            auto_filters=dict(inferred),
            applied_filters=applied,
            fallback_unfiltered=fell_back,
            hybrid_child_hits=child_hits,
            deduplicated_parents=len(candidates),
            reranked_kept=len(all_kept),
            reranked_dropped=max(0, len(candidates) - len(all_kept)),
            min_score=eff_min_score,
            results=final_results,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        retrieve_k: int | None = None,
        project: str | None = None,
        building: str | None = None,
        phase: str | None = None,
        auto_filter: bool = False,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Run hybrid search + cross-encoder reranking without LLM generation."""
        return self.retrieve_with_trace(
            query,
            top_k=top_k,
            retrieve_k=retrieve_k,
            project=project,
            building=building,
            phase=phase,
            auto_filter=auto_filter,
            min_score=min_score,
        ).results

    def run(
        self,
        query: str,
        *,
        top_k: int | None = None,
        retrieve_k: int | None = None,
        project: str | None = None,
        building: str | None = None,
        phase: str | None = None,
        auto_filter: bool = False,
        min_score: float | None = None,
    ) -> Answer:
        """Execute retrieval, reranking, and citation-grounded generation."""
        reranked = self.retrieve(
            query,
            top_k=top_k,
            retrieve_k=retrieve_k,
            project=project,
            building=building,
            phase=phase,
            auto_filter=auto_filter,
            min_score=min_score,
        )
        return self.generator.generate(query, reranked)


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    """Return `OpenAICompatibleLLM` when configured, else deterministic `ExtractiveStubLLM`."""
    cfg = settings or get_settings()
    if cfg.llm_base_url and cfg.llm_model:
        return OpenAICompatibleLLM(
            base_url=cfg.llm_base_url,
            model=cfg.llm_model,
            api_key=cfg.llm_api_key,
        )
    return ExtractiveStubLLM()


def build_rag_pipeline(
    *,
    qdrant_path: str | Path | None = None,
    device: str | None = None,
    settings: Settings | None = None,
) -> tuple[RAGPipeline, LocalHybridRepository]:
    """Build a ready-to-use `(RAGPipeline, LocalHybridRepository)` pair (caller should close repo)."""
    cfg = settings or get_settings()
    repo = LocalHybridRepository(path=str(qdrant_path or cfg.qdrant_path))
    embedder = BGEM3Embedder(
        model_name=cfg.embedding_model,
        device=device or cfg.device,
        batch_size=cfg.embedding_batch_size,
        max_length=cfg.embedding_max_length,
    )
    reranker = BGEM3Reranker(
        model_name=cfg.reranker_model,
        device=device or cfg.device,
        batch_size=cfg.reranker_batch_size,
        max_length=cfg.reranker_max_length,
        min_score=cfg.reranker_min_score,
    )
    pipeline = RAGPipeline(
        searcher=Searcher(embedder=embedder, repository=repo),
        reranker=reranker,
        generator=GroundedGenerator(llm=build_llm_client(cfg)),
        retrieve_k=cfg.retrieve_k,
        child_candidate_limit=cfg.child_candidate_limit,
        rerank_k=cfg.rerank_k,
    )
    return pipeline, repo
