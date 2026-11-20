"""Unified ingestion & ETL diagnostics CLI (`ingenium-ingest`)."""

import json
import logging
import sys
import time
from pathlib import Path

from rich.table import Table

from config import configure_logging, get_settings
from etl import ETLRunner, SharePointConnector
from scripts._cli import (
    add_common_args,
    build_parser,
    err_console,
    out_console,
    run_cli,
    suppress_hf_noise_early,
)


def _print_inventory(title: str, summary) -> None:
    ext_table = Table(title=f"Inventory: {title}", header_style="bold cyan")
    ext_table.add_column("Extension", style="cyan")
    ext_table.add_column("Count", justify="right", style="green")
    for ext, count in sorted(summary.extension_counts.items()):
        ext_table.add_row(ext, str(count))
    ext_table.add_row("TOTAL", str(summary.total_files), style="bold")
    out_console.print(ext_table)

    meta_table = Table(title="Inferred Metadata Preview (first 40 files)", header_style="bold cyan")
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
    out_console.print(meta_table)


def _print_report(processed_dir: Path, limitations_log: Path, *, as_json: bool = False) -> None:
    etl = ETLRunner(
        markdown_out_dir=processed_dir / "markdown",
        json_out_dir=processed_dir / "json",
        manifest_path=processed_dir / "manifest.json",
        limitations_report_path=limitations_log,
    )
    report = etl.get_summary_report()

    if as_json:
        payload = {
            "total_documents": report["total_documents"],
            "total_pages": report["total_pages"],
            "total_tables": report["total_tables"],
            "needs_ocr_count": report["needs_ocr_count"],
            "entries": [e.model_dump(mode="json") for e in report["entries"]],
            "limitations": [r.model_dump(mode="json") for r in report["limitations"]],
        }
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return

    table = Table(title="Extracted Documents Report", header_style="bold cyan")
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
    out_console.print(table)
    out_console.print(
        f"[bold]Totals:[/bold] {report['total_documents']} docs | "
        f"{report['total_pages']} pages | {report['total_tables']} tables | "
        f"{report['needs_ocr_count']} needs_ocr | {len(report['limitations'])} limitations"
    )
    if report["limitations"]:
        lim_table = Table(title="Logged Extraction Limitations", header_style="bold red")
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
        out_console.print(lim_table)


def _main() -> None:
    suppress_hf_noise_early()
    settings = get_settings()

    parser = build_parser(
        "Ingenium ETL extraction, diagnostics, and local Qdrant hybrid indexing CLI.",
        epilog=(
            "Examples:\n"
            "  ingenium-ingest --etl-dir data/processed\n"
            "  ingenium-ingest --raw-dir data/raw --extract-only\n"
            "  ingenium-ingest --from-sharepoint --folder-prefix General/05_Uitvoering --max-files 10\n"
            "  ingenium-ingest --report"
        ),
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
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=settings.processed_dir,
        help=f"Processed ParsedDoc directory (default: {settings.processed_dir}).",
    )
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
    add_common_args(parser, default_qdrant_path=settings.qdrant_path)

    args = parser.parse_args()
    configure_logging(
        verbose=args.verbose,
        quiet=args.quiet,
        json_mode=args.as_json,
        default_level=logging.INFO,
        disable_progress=args.quiet or args.as_json,
    )

    catalog = args.catalog or (settings.catalog_path if settings.catalog_path.exists() else None)

    if args.report:
        _print_report(args.processed_dir, settings.limitations_report_path, as_json=args.as_json)
        return

    if args.dry_run:
        etl = ETLRunner(catalog_path=catalog)
        if args.from_sharepoint:
            connector = SharePointConnector.from_env(json_out_dir=args.processed_dir / "json")
            summary = etl.scan_sharepoint_inventory(
                connector=connector, folder_prefix=args.folder_prefix
            )
            title = f"SharePoint ({connector.config.hostname}{connector.config.site_path})"
        elif args.raw_dir is not None:
            summary = etl.scan_inventory(args.raw_dir)
            title = str(args.raw_dir)
        else:
            parser.error("--dry-run requires --raw-dir or --from-sharepoint.")
            return

        if args.as_json:
            sys.stdout.write(
                json.dumps(
                    {
                        "source": title,
                        "total_files": summary.total_files,
                        "extension_counts": summary.extension_counts,
                    },
                    indent=2,
                )
                + "\n"
            )
        else:
            _print_inventory(title, summary)
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
            return

        elapsed = time.perf_counter() - t0
        if args.as_json:
            sys.stdout.write(
                json.dumps(
                    {
                        "elapsed_seconds": round(elapsed, 2),
                        "discovered_files": res.discovered_files,
                        "processed_files": res.processed_files,
                        "skipped_unchanged_files": res.skipped_unchanged_files,
                        "failed_files": res.failed_files,
                        "limitations_logged": res.limitations_logged,
                    },
                    indent=2,
                )
                + "\n"
            )
            return

        tbl = Table(title=f"ETL Summary ({elapsed:.1f}s)", header_style="bold cyan")
        tbl.add_column("Metric", style="cyan")
        tbl.add_column("Count", justify="right", style="green")
        tbl.add_row("Discovered", str(res.discovered_files))
        tbl.add_row("Processed", str(res.processed_files))
        tbl.add_row("Skipped (unchanged)", str(res.skipped_unchanged_files))
        tbl.add_row("Failed", str(res.failed_files))
        tbl.add_row("Limitations logged", str(res.limitations_logged))
        out_console.print(tbl)
        return

    from pipeline import IndexingPipeline
    from storage import LocalHybridRepository

    if not args.quiet and not args.as_json:
        err_console.print(f"[dim]Opening local Qdrant index at {args.qdrant_path}...[/dim]")

    with LocalHybridRepository(path=str(args.qdrant_path)) as repo:
        pipeline = IndexingPipeline(repository=repo, device=args.device)

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
    if args.as_json:
        sys.stdout.write(
            json.dumps(
                {
                    "elapsed_seconds": round(elapsed, 2),
                    "total_docs": stats.total_docs,
                    "indexed_docs": stats.indexed_docs,
                    "skipped_unchanged": stats.skipped_unchanged,
                    "skipped_empty": stats.skipped_empty,
                    "failed_docs": stats.failed_docs,
                    "new_parents": stats.total_parents,
                    "new_children": stats.total_children,
                    "qdrant": counts,
                },
                indent=2,
            )
            + "\n"
        )
        return

    summary_tbl = Table(title=f"Ingestion Complete ({elapsed:.1f}s)", header_style="bold cyan")
    summary_tbl.add_column("Metric", style="cyan")
    summary_tbl.add_column("Value", justify="right", style="green")
    summary_tbl.add_row("Documents seen", str(stats.total_docs))
    summary_tbl.add_row("Indexed", str(stats.indexed_docs))
    summary_tbl.add_row("Skipped (unchanged)", str(stats.skipped_unchanged))
    summary_tbl.add_row("Skipped (empty)", str(stats.skipped_empty))
    summary_tbl.add_row("Failed", str(stats.failed_docs))
    summary_tbl.add_row(
        "New parent / child chunks", f"{stats.total_parents} / {stats.total_children}"
    )
    summary_tbl.add_row(
        "Qdrant totals (docs / parents / children)",
        f"{counts['documents']} / {counts['parent_chunks']} / {counts['child_chunks']}",
    )
    out_console.print(summary_tbl)


def main() -> None:
    run_cli(_main)


if __name__ == "__main__":
    main()
