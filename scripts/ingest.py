"""Unified ingestion & ETL diagnostics CLI (`ingenium-ingest`).

Supports:
  - Indexing persisted `ParsedDoc` JSONs (`--etl-dir data/processed`)
  - Extracting & indexing local raw folders (`--raw-dir <dir>`)
  - Streaming & indexing directly from SharePoint (`--from-sharepoint [--folder-prefix ...] [--max-files N]`)
  - Metadata inventory preview (`--dry-run`)
  - Extraction & scanned-PDF/limitations diagnostics (`--report`)
"""

import argparse
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config import get_settings
from embedding import BGEM3Embedder
from etl import ETLRunner, SharePointConnector
from pipeline import IndexingPipeline
from storage import LocalHybridRepository


def _print_inventory(console: Console, title: str, summary) -> None:
    ext_table = Table(title=f"Inventory: {title}")
    ext_table.add_column("Extension", style="cyan")
    ext_table.add_column("Count", justify="right", style="green")
    for ext, count in sorted(summary.extension_counts.items()):
        ext_table.add_row(ext, str(count))
    ext_table.add_row("TOTAL", str(summary.total_files), style="bold")
    console.print(ext_table)

    meta_table = Table(title="Inferred Metadata Preview (first 40 files)")
    meta_table.add_column("Relative Path")
    meta_table.add_column("Project", style="cyan")
    meta_table.add_column("Building", style="magenta")
    meta_table.add_column("Phase", style="yellow")
    meta_table.add_column("Category", style="green")
    for item in summary.items[:40]:
        m = item.matched_metadata
        meta_table.add_row(
            item.relative_path,
            m.project or "-",
            m.building or "-",
            m.phase or "-",
            m.doc_category.value,
        )
    console.print(meta_table)


def _print_report(console: Console, processed_dir: Path, limitations_log: Path) -> None:
    etl = ETLRunner(
        markdown_out_dir=processed_dir / "markdown",
        json_out_dir=processed_dir / "json",
        manifest_path=processed_dir / "manifest.json",
        limitations_report_path=limitations_log,
    )
    report = etl.get_summary_report()
    table = Table(title="Extracted Documents Report")
    table.add_column("Doc ID", style="cyan")
    table.add_column("File Name")
    table.add_column("Project", style="green")
    table.add_column("Building", style="magenta")
    table.add_column("Phase", style="yellow")
    table.add_column("Sections", justify="right")
    table.add_column("OCR?", justify="center")
    for e in report["entries"]:
        table.add_row(
            e.doc_id,
            e.file_name,
            e.project or "-",
            e.building or "-",
            e.phase or "-",
            str(e.section_count),
            "YES" if e.needs_ocr else "-",
        )
    console.print(table)
    console.print(
        f"[bold]Totals:[/bold] {report['total_documents']} docs | "
        f"{report['total_pages']} pages | {report['total_tables']} tables | "
        f"{report['needs_ocr_count']} needs_ocr | {len(report['limitations'])} limitations"
    )
    if report["limitations"]:
        lim_table = Table(title="Logged Extraction Limitations")
        lim_table.add_column("File Name", style="red")
        lim_table.add_column("Type", style="yellow")
        lim_table.add_column("Confidence", justify="right")
        lim_table.add_column("Recommended Action")
        for rec in report["limitations"]:
            lim_table.add_row(
                rec.file_name,
                rec.limitation_type,
                f"{rec.extraction_confidence:.2f}",
                rec.recommended_action,
            )
        console.print(lim_table)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    console = Console()

    parser = argparse.ArgumentParser(
        description="Ingenium ETL extraction, diagnostics, and local Qdrant hybrid indexing CLI."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--etl-dir", type=Path, help="Index persisted ParsedDoc JSONs (e.g. data/processed)."
    )
    mode.add_argument(
        "--raw-dir", type=Path, help="Extract local raw folder and index into Qdrant."
    )
    mode.add_argument(
        "--from-sharepoint", action="store_true", help="Stream from SharePoint and index."
    )
    mode.add_argument(
        "--report", action="store_true", help="Print manifest & scanned-PDF/limitations report."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview file inventory & metadata without extracting.",
    )
    parser.add_argument(
        "--extract-only", action="store_true", help="Run ETL extraction without embedding/indexing."
    )
    parser.add_argument(
        "--catalog", type=Path, default=None, help="Optional project catalog (.csv/.xlsx)."
    )
    parser.add_argument("--processed-dir", type=Path, default=settings.processed_dir)
    parser.add_argument("--qdrant-path", type=Path, default=settings.qdrant_path)
    parser.add_argument(
        "--folder-prefix", type=str, default=None, help="SharePoint subfolder prefix."
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Cap SharePoint files streamed."
    )
    parser.add_argument(
        "--device", type=str, default=settings.device, help="Device override (cuda/mps/cpu)."
    )
    parser.add_argument("--force", action="store_true", help="Force re-extraction/re-indexing.")
    parser.add_argument(
        "--keep-noisy-callouts", action="store_true", help="Disable diagram callout quarantine."
    )

    args = parser.parse_args()
    catalog = args.catalog or (settings.catalog_path if settings.catalog_path.exists() else None)

    if args.report:
        _print_report(console, args.processed_dir, settings.limitations_report_path)
        return

    if args.dry_run:
        etl = ETLRunner(catalog_path=catalog)
        if args.from_sharepoint:
            connector = SharePointConnector.from_env(json_out_dir=args.processed_dir / "json")
            summary = etl.scan_sharepoint_inventory(
                connector=connector, folder_prefix=args.folder_prefix
            )
            _print_inventory(
                console,
                f"SharePoint ({connector.config.hostname}{connector.config.site_path})",
                summary,
            )
        elif args.raw_dir is not None:
            _print_inventory(console, str(args.raw_dir), etl.scan_inventory(args.raw_dir))
        else:
            parser.error("--dry-run requires --raw-dir or --from-sharepoint.")
        return

    t0 = time.perf_counter()

    if args.extract_only:
        etl = ETLRunner(
            catalog_path=catalog,
            markdown_out_dir=args.processed_dir / "markdown",
            json_out_dir=args.processed_dir / "json",
            manifest_path=args.processed_dir / "manifest.json",
            drop_noisy_callouts=not args.keep_noisy_callouts,
        )
        if args.from_sharepoint:
            res = etl.run_sharepoint(
                force=args.force, max_files=args.max_files, folder_prefix=args.folder_prefix
            )
        elif args.raw_dir is not None:
            res = etl.run_local(args.raw_dir, force=args.force)
        else:
            parser.error("--extract-only requires --raw-dir or --from-sharepoint.")
        console.print(
            f"ETL done in {time.perf_counter() - t0:.1f}s | discovered={res.discovered_files} "
            f"processed={res.processed_files} skipped={res.skipped_unchanged_files} "
            f"failed={res.failed_files} limitations={res.limitations_logged}"
        )
        return

    with LocalHybridRepository(path=str(args.qdrant_path)) as repo:
        embedder = BGEM3Embedder(model_name=settings.embedding_model, device=args.device)
        pipeline = IndexingPipeline(repository=repo, embedder=embedder)

        if args.etl_dir is not None:
            stats = pipeline.ingest_processed_dir(args.etl_dir, force=args.force)
        elif args.from_sharepoint:
            stats = pipeline.ingest_sharepoint(
                catalog_path=catalog,
                processed_dir=args.processed_dir,
                force_etl=args.force,
                force_index=args.force,
                max_files=args.max_files,
                folder_prefix=args.folder_prefix,
                drop_noisy_callouts=not args.keep_noisy_callouts,
            )
        else:
            stats = pipeline.ingest_raw_folder(
                args.raw_dir,
                catalog_path=catalog,
                processed_dir=args.processed_dir,
                force_etl=args.force,
                force_index=args.force,
                drop_noisy_callouts=not args.keep_noisy_callouts,
            )

        counts = repo.collection_counts()

    elapsed = time.perf_counter() - t0
    console.print(
        f"Done in {elapsed:.1f}s | docs={stats.total_docs} indexed={stats.indexed_docs} "
        f"skipped_unchanged={stats.skipped_unchanged} skipped_empty={stats.skipped_empty} "
        f"failed={stats.failed_docs} | new_parents={stats.total_parents} new_children={stats.total_children} "
        f"| qdrant={counts}"
    )


if __name__ == "__main__":
    main()
