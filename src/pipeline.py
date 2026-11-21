"""Thin top-level orchestrator wiring isolated stages via canonical `domain` contracts.

This is the ONLY library module that imports across stages (`etl`, `chunking`,
`embedding`, `storage`, `retrieval`, `generation`). Every stage remains strictly
isolated and testable in complete independence.
"""

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from chunking import ParentChildChunker
from config import (
    Settings,
    SharePointSiteSpec,
    detect_device,
    get_settings,
    load_sharepoint_sites,
)
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


ProgressCallback = Callable[[str, int, int, str], None]


@dataclass
class IndexingStats:
    """Execution summary for an `IndexingPipeline` run."""

    discovered_files: int = 0
    total_docs: int = 0
    indexed_docs: int = 0
    skipped_unchanged: int = 0
    skipped_empty: int = 0
    scanned_empty_docs: int = 0
    dropped_noisy_callouts: int = 0
    failed_docs: int = 0
    unauthorized_count: int = 0
    unauthorized_targets: list[str] = field(default_factory=list)
    total_parents: int = 0
    total_children: int = 0
    stage_timings: dict[str, float] = field(default_factory=dict)


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

    def _get_embedder(self, on_progress: ProgressCallback | None = None) -> ChildEmbedder:
        if self._embedder is None:
            dev = detect_device(self._device or self._settings.device)
            if on_progress:
                on_progress(
                    "model_load_start",
                    0,
                    0,
                    f"Loading {self._settings.embedding_model} on {dev}...",
                )
            t0 = time.perf_counter()
            self._embedder = BGEM3Embedder(
                model_name=self._settings.embedding_model,
                device=dev,
                batch_size=self._settings.embedding_batch_size,
                max_length=self._settings.embedding_max_length,
            )
            if on_progress:
                on_progress(
                    "model_load_done",
                    0,
                    0,
                    f"{self._settings.embedding_model} ready on {dev} in {time.perf_counter() - t0:.1f}s",
                )
        return self._embedder

    @property
    def embedder(self) -> ChildEmbedder:
        """Lazily load `BGEM3Embedder` only when at least one chunk actually needs embedding."""
        return self._get_embedder()

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
        on_progress: ProgressCallback | None = None,
        stage_offset: int = 0,
        total_stages: int = 2,
    ) -> IndexingStats:
        """Index `ParsedDoc`s into Qdrant using length-sorted cross-document batch embedding."""
        t_start = time.perf_counter()
        docs_list = list(documents)
        stats = IndexingStats()

        # --- Chunking stage ---
        chunk_stage = stage_offset + 1
        if on_progress:
            on_progress("stage_start", chunk_stage, total_stages, "Chunking")
        t_chunk0 = time.perf_counter()
        pending_bundles: list[tuple[ParsedDoc, list[ParentChunk], list[ChildChunk]]] = []
        if on_progress and docs_list:
            on_progress("bar_start", 0, len(docs_list), "Chunking documents")

        for idx, doc in enumerate(docs_list, start=1):
            stats.total_docs += 1
            stats.dropped_noisy_callouts += int(
                doc.metadata.extra.get("dropped_noisy_callouts") or 0
            )
            try:
                if not force and not self.repository.needs_indexing(
                    doc.document_id, doc.content_hash
                ):
                    stats.skipped_unchanged += 1
                    logger.debug(
                        "Skipping unchanged '%s' (%s).", doc.metadata.source_path, doc.document_id
                    )
                    if on_progress:
                        on_progress(
                            "bar_advance",
                            idx,
                            len(docs_list),
                            f"unchanged {Path(doc.metadata.source_path).name}",
                        )
                    continue

                parents, children = self.chunker.chunk(doc)
                if not parents or not children:
                    stats.skipped_empty += 1
                    if doc.metadata.extra.get("is_scanned") or doc.metadata.extra.get("needs_ocr"):
                        stats.scanned_empty_docs += 1
                    logger.warning(
                        "Skipping '%s' (%s): 0 chunks generated.",
                        doc.metadata.source_path,
                        doc.document_id,
                    )
                    if on_progress:
                        on_progress(
                            "bar_advance",
                            idx,
                            len(docs_list),
                            f"empty {Path(doc.metadata.source_path).name}",
                        )
                    continue

                pending_bundles.append((doc, parents, children))
            except Exception:
                stats.failed_docs += 1
                logger.exception(
                    "Failed to prepare '%s' (%s).", doc.metadata.source_path, doc.document_id
                )
            if on_progress:
                on_progress("bar_advance", idx, len(docs_list), Path(doc.metadata.source_path).name)

        if on_progress and docs_list:
            on_progress("bar_done", len(docs_list), len(docs_list), "Chunking documents")
        chunk_elapsed = time.perf_counter() - t_chunk0
        stats.stage_timings["chunking"] = round(chunk_elapsed, 2)
        total_pending_children = sum(len(c) for _, _, c in pending_bundles)
        if on_progress:
            on_progress(
                "stage_done",
                chunk_stage,
                total_stages,
                f"Chunking done in {chunk_elapsed:.1f}s "
                f"({len(pending_bundles)} doc(s) queued, {total_pending_children} children, "
                f"{stats.skipped_unchanged} unchanged)",
            )

        # --- Hybrid Embedding & Qdrant Indexing stage ---
        embed_stage = stage_offset + 2
        eff_dev = getattr(self._embedder, "device", None) or detect_device(
            self._device or self._settings.device
        )
        if on_progress:
            on_progress(
                "stage_start",
                embed_stage,
                total_stages,
                f"Hybrid Embedding & Qdrant Indexing (device={eff_dev})",
            )
        t_embed_stage = time.perf_counter()
        embedded_so_far = 0
        if on_progress and total_pending_children:
            on_progress("bar_start", 0, total_pending_children, "Embedding & indexing chunks")

        buffer: list[tuple[ParsedDoc, list[ParentChunk], list[ChildChunk]]] = []
        buffered_children = 0

        def flush_buffer() -> None:
            nonlocal buffered_children, embedded_so_far
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
                embedder = self._get_embedder(on_progress=on_progress)
                sorted_batch = embedder.embed_children(sorted_children)

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

            embedded_so_far += len(flat_children)
            if on_progress and total_pending_children:
                on_progress(
                    "bar_advance",
                    embedded_so_far,
                    total_pending_children,
                    f"{len(buffer)} doc(s) upserted",
                )
            buffer.clear()
            buffered_children = 0

        for bundle in pending_bundles:
            buffer.append(bundle)
            buffered_children += len(bundle[2])
            if buffered_children >= self.embed_batch_buffer_size:
                flush_buffer()

        flush_buffer()
        if on_progress and total_pending_children:
            on_progress(
                "bar_done",
                total_pending_children,
                total_pending_children,
                "Embedding & indexing chunks",
            )
        embed_elapsed = time.perf_counter() - t_embed_stage
        stats.stage_timings["embedding_and_indexing"] = round(embed_elapsed, 2)
        if on_progress:
            on_progress(
                "stage_done",
                embed_stage,
                total_stages,
                f"Embedding & Indexing done in {embed_elapsed:.1f}s "
                f"({stats.indexed_docs} indexed, {stats.total_parents} parents, "
                f"{stats.total_children} children)",
            )
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
        on_progress: ProgressCallback | None = None,
        stage_offset: int = 0,
        total_stages: int = 2,
    ) -> IndexingStats:
        """Load persisted `ParsedDoc` JSONs from `processed_dir` and index them."""
        docs = load_processed_docs(
            processed_dir,
            use_manifest=use_manifest,
            dedupe_by_hash=dedupe_by_hash,
        )
        return self.index_documents(
            docs,
            force=force,
            on_progress=on_progress,
            stage_offset=stage_offset,
            total_stages=total_stages,
        )

    def ingest_raw_folder(
        self,
        raw_dir: str | Path,
        *,
        catalog_path: str | Path | None = None,
        processed_dir: str | Path | None = None,
        force_etl: bool = False,
        force_index: bool = False,
        drop_noisy_callouts: bool = True,
        workers: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> IndexingStats:
        """Run local ETL extraction into `processed_dir`, then index all valid `ParsedDoc`s."""
        cfg = get_settings()
        out_base = Path(processed_dir) if processed_dir is not None else cfg.processed_dir
        etl = ETLRunner(
            catalog_path=catalog_path,
            markdown_out_dir=out_base / "markdown",
            json_out_dir=out_base / "json",
            manifest_path=out_base / "manifest.json",
            drop_noisy_callouts=drop_noisy_callouts,
        )
        if on_progress:
            on_progress("stage_start", 1, 4, "Discovery & Local Sync")
            on_progress("stage_done", 1, 4, f"Local source ready ({raw_dir})")
            on_progress("stage_start", 2, 4, "ETL Extraction")
        t_etl = time.perf_counter()
        etl_res = etl.run_local(
            raw_dir,
            force=force_etl,
            workers=workers if workers is not None else cfg.etl_workers,
            on_progress=on_progress,
        )
        etl_elapsed = time.perf_counter() - t_etl
        if on_progress:
            on_progress(
                "stage_done",
                2,
                4,
                f"ETL Extraction done in {etl_elapsed:.1f}s "
                f"({etl_res.processed_files} parsed, {etl_res.scanned_empty_files} scanned/empty, "
                f"{etl_res.skipped_unchanged_files} unchanged)",
            )
        stats = self.ingest_processed_dir(
            out_base,
            force=force_index,
            on_progress=on_progress,
            stage_offset=2,
            total_stages=4,
        )
        stats.discovered_files = etl_res.discovered_files
        stats.scanned_empty_docs = max(stats.scanned_empty_docs, etl_res.scanned_empty_files)
        stats.dropped_noisy_callouts = max(
            stats.dropped_noisy_callouts, etl_res.dropped_noisy_callouts
        )
        stats.stage_timings = {"etl": round(etl_elapsed, 2), **stats.stage_timings}
        return stats

    def ingest_sharepoint(
        self,
        *,
        sites: Sequence[SharePointSiteSpec] | None = None,
        catalog_path: str | Path | None = None,
        processed_dir: str | Path | None = None,
        force_etl: bool = False,
        force_index: bool = False,
        max_files: int | None = None,
        folder_prefix: str | None = None,
        drop_noisy_callouts: bool = True,
        workers: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> IndexingStats:
        """Stream files from one or more SharePoint sites via ETL and incrementally index into Qdrant."""
        cfg = get_settings()
        out_base = Path(processed_dir) if processed_dir is not None else cfg.processed_dir
        eff_workers = workers if workers is not None else cfg.etl_workers
        site_list = (
            list(sites)
            if sites is not None
            else load_sharepoint_sites(folder_prefix=folder_prefix, settings=cfg)
        )
        if not site_list:
            raise ValueError(
                "No SharePoint sites configured. Set SHAREPOINT_SITE_PATH / SHAREPOINT_SITE_PATHS "
                "in .env, or pass --site / --sites-file."
            )

        etl = ETLRunner(
            catalog_path=catalog_path,
            markdown_out_dir=out_base / "markdown",
            json_out_dir=out_base / "json",
            manifest_path=out_base / "manifest.json",
            drop_noisy_callouts=drop_noisy_callouts,
        )
        base_connector = SharePointConnector.from_env(json_out_dir=out_base / "json")
        namespace_by_site = len(site_list) > 1

        if on_progress:
            on_progress(
                "stage_start",
                1,
                4,
                f"SharePoint Discovery & Sync ({len(site_list)} site(s), workers={eff_workers})",
            )

        t_etl_total = time.perf_counter()
        total_discovered = 0
        total_parsed = 0
        total_skipped_etl = 0
        total_scanned_empty = 0
        total_dropped_callouts = 0
        total_unauth = 0
        unauth_targets: list[str] = []

        if on_progress:
            on_progress("stage_start", 2, 4, "SharePoint Streaming & ETL Extraction")

        for s_idx, spec in enumerate(site_list, start=1):
            site_sp = base_connector.for_site(spec)
            logger.info(
                "Syncing SharePoint site [%d/%d]: %s%s (project=%s)",
                s_idx,
                len(site_list),
                spec.hostname,
                spec.site_path,
                spec.project_code or "auto",
            )
            etl_res = etl.run_sharepoint(
                connector=site_sp,
                force=force_etl,
                max_files=max_files,
                folder_prefix=folder_prefix or spec.folder_prefix,
                workers=eff_workers,
                project_override=spec.project_code,
                namespace_by_site=namespace_by_site,
                on_progress=on_progress,
            )
            total_discovered += etl_res.discovered_files
            total_parsed += etl_res.processed_files
            total_skipped_etl += etl_res.skipped_unchanged_files
            total_scanned_empty += etl_res.scanned_empty_files
            total_dropped_callouts += etl_res.dropped_noisy_callouts
            total_unauth += etl_res.unauthorized_count
            unauth_targets.extend(etl_res.unauthorized_targets)

        etl_elapsed = time.perf_counter() - t_etl_total
        if on_progress:
            on_progress(
                "stage_done",
                2,
                4,
                f"SharePoint ETL done in {etl_elapsed:.1f}s "
                f"({total_parsed} parsed, {total_scanned_empty} scanned/empty, "
                f"{total_skipped_etl} unchanged, {total_unauth} unauthorized)",
            )

        stats = self.ingest_processed_dir(
            out_base,
            force=force_index,
            on_progress=on_progress,
            stage_offset=2,
            total_stages=4,
        )
        stats.discovered_files = total_discovered
        stats.scanned_empty_docs = max(stats.scanned_empty_docs, total_scanned_empty)
        stats.dropped_noisy_callouts = max(stats.dropped_noisy_callouts, total_dropped_callouts)
        stats.unauthorized_count = total_unauth
        stats.unauthorized_targets = unauth_targets
        stats.stage_timings = {"etl": round(etl_elapsed, 2), **stats.stage_timings}
        return stats


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
    on_progress: ProgressCallback | None = None,
) -> tuple[RAGPipeline, LocalHybridRepository]:
    """Build a ready-to-use `(RAGPipeline, LocalHybridRepository)` pair (caller should close repo)."""
    cfg = settings or get_settings()
    eff_dev = detect_device(device or cfg.device)
    repo = LocalHybridRepository(path=str(qdrant_path or cfg.qdrant_path))

    if on_progress:
        on_progress("model_load_start", 0, 0, f"Loading {cfg.embedding_model} on {eff_dev}...")
    t0 = time.perf_counter()
    embedder = BGEM3Embedder(
        model_name=cfg.embedding_model,
        device=eff_dev,
        batch_size=cfg.embedding_batch_size,
        max_length=cfg.embedding_max_length,
    )
    if on_progress:
        on_progress(
            "model_load_done",
            0,
            0,
            f"{cfg.embedding_model} ready on {eff_dev} in {time.perf_counter() - t0:.1f}s",
        )

    if on_progress:
        on_progress("model_load_start", 0, 0, f"Loading {cfg.reranker_model} on {eff_dev}...")
    t1 = time.perf_counter()
    reranker = BGEM3Reranker(
        model_name=cfg.reranker_model,
        device=eff_dev,
        batch_size=cfg.reranker_batch_size,
        max_length=cfg.reranker_max_length,
        min_score=cfg.reranker_min_score,
    )
    if on_progress:
        on_progress(
            "model_load_done",
            0,
            0,
            f"{cfg.reranker_model} ready on {eff_dev} in {time.perf_counter() - t1:.1f}s",
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
