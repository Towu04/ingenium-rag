"""CLI entry point to ingest ETL-processed JSONs or raw engineering folders into local Qdrant."""

import argparse
import logging
from pathlib import Path
import time

from config import get_settings
from pipelines.indexing_pipeline import IndexingPipeline, IndexingStats
from storage.embedder import BGEM3Embedder
from storage.repository import LocalHybridRepository


def ingest_etl_dir(
    etl_dir: Path,
    qdrant_path: Path,
    *,
    force: bool = False,
    drop_noisy_callouts: bool = True,
    device: str | None = None,
) -> IndexingStats:
    repo = LocalHybridRepository(path=str(qdrant_path))
    embedder = BGEM3Embedder(device=device)
    pipeline = IndexingPipeline(repository=repo, embedder=embedder)
    return pipeline.ingest_from_etl_dir(
        etl_dir,
        force=force,
        drop_noisy_callouts=drop_noisy_callouts,
    )


def ingest_raw_dir(
    raw_dir: Path,
    qdrant_path: Path,
    *,
    catalog_path: Path | None = None,
    processed_dir: Path = Path("data/processed"),
    force: bool = False,
    drop_noisy_callouts: bool = True,
    device: str | None = None,
) -> IndexingStats:
    repo = LocalHybridRepository(path=str(qdrant_path))
    embedder = BGEM3Embedder(device=device)
    pipeline = IndexingPipeline(repository=repo, embedder=embedder)
    return pipeline.ingest_raw_folder(
        raw_dir,
        catalog_path=catalog_path,
        processed_dir=processed_dir,
        force_etl=force,
        force_index=force,
        drop_noisy_callouts=drop_noisy_callouts,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Index engineering documents into the local Qdrant hybrid vector store."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--etl-dir",
        type=Path,
        help="Path to ETL processed directory containing json/ and manifest.json (e.g. data/processed).",
    )
    source_group.add_argument(
        "--raw-dir",
        type=Path,
        help="Path to raw engineering documents folder (runs ETLPipeline before indexing).",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Optional project catalog (.csv/.xlsx) when using --raw-dir.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=settings.qdrant_path,
        help=f"Local Qdrant storage path (default: {settings.qdrant_path}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=settings.device,
        help="Torch device override (cuda, mps, cpu).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even when content_hash is unchanged.",
    )
    parser.add_argument(
        "--keep-noisy-callouts",
        action="store_true",
        help="Keep low-readability diagram_callout blocks instead of quarantining them.",
    )

    args = parser.parse_args()
    t0 = time.perf_counter()

    if args.etl_dir is not None:
        stats = ingest_etl_dir(
            args.etl_dir,
            args.qdrant_path,
            force=args.force,
            drop_noisy_callouts=not args.keep_noisy_callouts,
            device=args.device,
        )
    else:
        stats = ingest_raw_dir(
            args.raw_dir,
            args.qdrant_path,
            catalog_path=args.catalog,
            processed_dir=settings.processed_dir,
            force=args.force,
            drop_noisy_callouts=not args.keep_noisy_callouts,
            device=args.device,
        )

    elapsed = time.perf_counter() - t0
    print(
        f"Done in {elapsed:.1f}s | docs={stats.total_docs} indexed={stats.indexed_docs} "
        f"skipped_unchanged={stats.skipped_unchanged} skipped_empty={stats.skipped_empty} "
        f"failed={stats.failed_docs} | parents={stats.total_parents} children={stats.total_children}"
    )


if __name__ == "__main__":
    main()
