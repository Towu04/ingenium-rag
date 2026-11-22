"""Unified ingestion & ETL diagnostics CLI (`ingenium-ingest`)."""

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from rich.table import Table

from config import get_settings, load_sharepoint_sites
from etl import ETLRunner, SharePointAuthError, SharePointConnector
from scripts._cli import (
    PipelineProgress,
    add_common_args,
    build_parser,
    err_console,
    format_duration,
    out_console,
    print_kv_table,
    run_cli,
    setup_cli_logging,
    suppress_hf_noise_early,
)


def _print_inventory(title: str, summary, *, preview_limit: int = 40) -> None:
    ext_table = Table(title=f"Inventory: {title}", header_style="bold cyan")
    ext_table.add_column("Extension", style="cyan")
    ext_table.add_column("Count", justify="right", style="green")
    for ext, count in sorted(summary.extension_counts.items()):
        ext_table.add_row(ext, f"{count:,}")
    ext_table.add_row("TOTAL", f"{summary.total_files:,}", style="bold")
    out_console.print(ext_table)

    items = summary.items if preview_limit <= 0 else summary.items[:preview_limit]
    showing = (
        f"all {len(items):,} files"
        if len(items) == len(summary.items)
        else f"first {len(items):,} of {len(summary.items):,} files"
    )
    out_console.print()
    meta_table = Table(title=f"Inferred Metadata Preview ({showing})", header_style="bold cyan")
    meta_table.add_column("Relative Path", no_wrap=True, overflow="ellipsis")
    meta_table.add_column("Project", style="cyan", no_wrap=True)
    meta_table.add_column("Building", style="magenta", no_wrap=True, overflow="ellipsis")
    meta_table.add_column("Phase", style="yellow", no_wrap=True)
    meta_table.add_column("Category", style="green", no_wrap=True)
    for item in items:
        m = item.matched_metadata
        meta_table.add_row(
            item.relative_path,
            m.project or "-",
            m.building or "-",
            m.phase or "-",
            m.doc_category.value,
        )
    out_console.print(meta_table)
    remaining = len(summary.items) - len(items)
    if remaining > 0:
        out_console.print(
            f"[dim]... and {remaining:,} more files (pass --limit 0 to list all).[/dim]"
        )


def _print_report(
    processed_dir: Path,
    limitations_log: Path,
    *,
    as_json: bool = False,
    limit: int = 20,
) -> None:
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

    entries = report["entries"]
    limitations = report["limitations"]
    lim_by_type = Counter(rec.limitation_type for rec in limitations)
    proj_counts = Counter(e.project or "(unspecified)" for e in entries)
    unauth_count = lim_by_type.get("unauthorized_access", 0)

    print_kv_table(
        "Extraction & Audit Overview",
        [
            ("Extracted documents", f"{report['total_documents']:,}"),
            ("Total pages", f"{report['total_pages']:,}"),
            ("Total tables", f"{report['total_tables']:,}"),
            ("Needs OCR (scanned/empty)", f"{report['needs_ocr_count']:,}"),
            ("Unauthorized (401/403)", f"{unauth_count:,}"),
            ("Total limitations logged", f"{len(limitations):,}"),
        ],
    )

    if proj_counts:
        out_console.print()
        proj_table = Table(title="Documents by Project", header_style="bold cyan")
        proj_table.add_column("Project", style="cyan", no_wrap=True)
        proj_table.add_column("Docs", justify="right", style="green")
        proj_table.add_column("Share", justify="right", style="dim")
        total_docs = max(len(entries), 1)
        shown_projects = proj_counts.most_common(15 if limit > 0 else None)
        for proj, cnt in shown_projects:
            label = f"[dim]{proj}[/dim]" if proj == "(unspecified)" else proj
            proj_table.add_row(label, f"{cnt:,}", f"{(cnt / total_docs) * 100:.1f}%")
        out_console.print(proj_table)
        if len(proj_counts) > len(shown_projects):
            out_console.print(
                f"[dim]... and {len(proj_counts) - len(shown_projects):,} more projects.[/dim]"
            )

    if lim_by_type:
        out_console.print()
        lim_summary = Table(title="Limitations Breakdown", header_style="bold yellow")
        lim_summary.add_column("Limitation Type", style="yellow", no_wrap=True)
        lim_summary.add_column("Count", justify="right", style="red")
        for ltype, cnt in lim_by_type.most_common():
            lim_summary.add_row(ltype, f"{cnt:,}")
        out_console.print(lim_summary)

    shown_entries = entries if limit <= 0 else entries[:limit]
    if shown_entries:
        out_console.print()
        entry_title = (
            f"Extracted Documents (all {len(entries):,})"
            if len(shown_entries) == len(entries)
            else f"Extracted Documents (showing {len(shown_entries):,} of {len(entries):,})"
        )
        table = Table(title=entry_title, header_style="bold cyan")
        table.add_column("Doc ID", style="cyan", no_wrap=True)
        table.add_column("File Name", no_wrap=True, overflow="ellipsis", max_width=28)
        table.add_column("Project", style="green", no_wrap=True)
        table.add_column("Phase", style="yellow", no_wrap=True, overflow="ellipsis", max_width=14)
        table.add_column("Sec.", justify="right", no_wrap=True)
        table.add_column("OCR?", justify="center", no_wrap=True)
        for e in shown_entries:
            table.add_row(
                e.doc_id[:10],
                e.file_name,
                e.project or "-",
                e.phase or "-",
                f"{e.section_count:,}",
                "[yellow]YES[/yellow]" if e.needs_ocr else "[dim]-[/dim]",
            )
        out_console.print(table)
        rem_docs = len(entries) - len(shown_entries)
        if rem_docs > 0:
            out_console.print(
                f"[dim]... and {rem_docs:,} more documents (run `make report LIMIT=0` to show all).[/dim]"
            )

    shown_lims = limitations if limit <= 0 else limitations[:limit]
    if shown_lims:
        out_console.print()
        lim_title = (
            f"Logged Extraction Limitations (all {len(limitations):,})"
            if len(shown_lims) == len(limitations)
            else f"Logged Extraction Limitations (showing {len(shown_lims):,} of {len(limitations):,})"
        )
        lim_table = Table(title=lim_title, header_style="bold red")
        lim_table.add_column(
            "File Name", style="red", no_wrap=True, overflow="ellipsis", max_width=24
        )
        lim_table.add_column(
            "Type", style="yellow", no_wrap=True, overflow="ellipsis", max_width=22
        )
        lim_table.add_column("Conf.", justify="right", no_wrap=True)
        lim_table.add_column("Recommended Action", no_wrap=True, overflow="ellipsis", max_width=28)
        for rec in shown_lims:
            lim_table.add_row(
                rec.file_name,
                rec.limitation_type,
                f"{rec.extraction_confidence:.2f}",
                rec.recommended_action,
            )
        out_console.print(lim_table)
        rem_lims = len(limitations) - len(shown_lims)
        if rem_lims > 0:
            out_console.print(
                f"[dim]... and {rem_lims:,} more limitations "
                f"(inspect {limitations_log} or run `make report LIMIT=0`).[/dim]"
            )


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
        "--site",
        type=str,
        default=None,
        help="Target a single SharePoint site URL or /sites/<slug> path.",
    )
    parser.add_argument(
        "--sites-file",
        type=Path,
        default=None,
        help="TSV/CSV file of SharePoint sites (project_code<TAB>url).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.etl_workers,
        help=f"Concurrent workers for SharePoint sync/ETL (default: {settings.etl_workers}).",
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Cap SharePoint files streamed per site."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max preview rows for --report / --dry-run (default: 20; 0 = show all).",
    )
    parser.add_argument(
        "--device", type=str, default=settings.device, help="Device override (cuda/mps/cpu)."
    )
    parser.add_argument("--force", action="store_true", help="Force re-extraction/re-indexing.")
    parser.add_argument(
        "--keep-noisy-callouts", action="store_true", help="Disable diagram callout quarantine."
    )
    parser.add_argument(
        "--include-archives",
        action="store_true",
        help="Include archive/draft folders (oud, old, archief, draft) that are skipped by default.",
    )
    add_common_args(parser, default_qdrant_path=settings.qdrant_path)

    args = parser.parse_args()
    setup_cli_logging(
        verbose=args.verbose,
        quiet=args.quiet,
        json_mode=args.as_json,
        default_level=logging.WARNING,
        disable_progress=args.quiet or args.as_json,
    )

    catalog = args.catalog or (settings.catalog_path if settings.catalog_path.exists() else None)

    if args.report:
        _print_report(
            args.processed_dir,
            settings.limitations_report_path,
            as_json=args.as_json,
            limit=args.limit,
        )
        return

    if args.dry_run:
        etl = ETLRunner(catalog_path=catalog)
        if args.from_sharepoint:
            sites = load_sharepoint_sites(
                sites_file=args.sites_file,
                site_override=args.site,
                folder_prefix=args.folder_prefix,
                settings=settings,
            )
            base_sp = SharePointConnector.from_env(json_out_dir=args.processed_dir / "json")
            all_ext: dict[str, int] = {}
            total_files = 0
            unauthorized_sites: list[str] = []
            site_rows: list[tuple[str, int, int, int, int, str]] = []
            for spec in sites:
                site_label = f"{spec.hostname}{spec.site_path}"
                try:
                    summary = etl.scan_sharepoint_inventory(
                        connector=base_sp.for_site(spec),
                        folder_prefix=args.folder_prefix or spec.folder_prefix,
                    )
                    total_files += summary.total_files
                    for ext, cnt in summary.extension_counts.items():
                        all_ext[ext] = all_ext.get(ext, 0) + cnt
                    site_rows.append(
                        (
                            site_label,
                            summary.total_files,
                            summary.extension_counts.get(".pdf", 0),
                            summary.extension_counts.get(".docx", 0),
                            summary.extension_counts.get(".xlsx", 0),
                            "[green]OK[/green]",
                        )
                    )
                    if not args.as_json:
                        _print_inventory(
                            f"SharePoint ({site_label})", summary, preview_limit=args.limit
                        )
                except SharePointAuthError as auth_err:
                    unauthorized_sites.append(site_label)
                    site_rows.append((site_label, 0, 0, 0, 0, "[red]401/403[/red]"))
                    err_console.print(f"[yellow]Skipping unauthorized site:[/yellow] {auth_err}")
            if args.as_json:
                sys.stdout.write(
                    json.dumps(
                        {
                            "sites": [f"{s.hostname}{s.site_path}" for s in sites],
                            "total_files": total_files,
                            "extension_counts": all_ext,
                            "unauthorized_sites": unauthorized_sites,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            elif len(sites) > 1:
                out_console.print()
                rollup = Table(
                    title=f"Multi-Site Dry-Run Rollup ({len(sites)} sites)",
                    header_style="bold cyan",
                )
                rollup.add_column("Site", style="cyan", no_wrap=True, overflow="ellipsis")
                rollup.add_column("Files", justify="right", style="green")
                rollup.add_column("PDF", justify="right")
                rollup.add_column("DOCX", justify="right")
                rollup.add_column("XLSX", justify="right")
                rollup.add_column("Status", justify="center")
                for s_lbl, n_all, n_pdf, n_docx, n_xlsx, status in site_rows:
                    rollup.add_row(
                        s_lbl,
                        f"{n_all:,}",
                        f"{n_pdf:,}",
                        f"{n_docx:,}",
                        f"{n_xlsx:,}",
                        status,
                    )
                rollup.add_section()
                rollup.add_row(
                    "[bold]TOTAL[/bold]",
                    f"[bold]{total_files:,}[/bold]",
                    f"[bold]{all_ext.get('.pdf', 0):,}[/bold]",
                    f"[bold]{all_ext.get('.docx', 0):,}[/bold]",
                    f"[bold]{all_ext.get('.xlsx', 0):,}[/bold]",
                    f"[dim]{len(sites) - len(unauthorized_sites)}/{len(sites)} OK[/dim]",
                )
                out_console.print(rollup)
            return
        elif args.raw_dir is not None:
            summary = etl.scan_inventory(args.raw_dir)
            title = str(args.raw_dir)
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
                _print_inventory(title, summary, preview_limit=args.limit)
            return
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
            skip_archive_folders=False if args.include_archives else None,
        )
        with PipelineProgress(quiet=args.quiet, json_mode=args.as_json) as prog:
            if args.from_sharepoint:
                sites = load_sharepoint_sites(
                    sites_file=args.sites_file,
                    site_override=args.site,
                    folder_prefix=args.folder_prefix,
                    settings=settings,
                )
                base_sp = SharePointConnector.from_env(json_out_dir=args.processed_dir / "json")
                ns_by_site = len(sites) > 1
                from etl import ETLRunResult

                res = ETLRunResult()
                for spec in sites:
                    site_res = etl.run_sharepoint(
                        connector=base_sp.for_site(spec),
                        force=args.force,
                        max_files=args.max_files,
                        folder_prefix=args.folder_prefix or spec.folder_prefix,
                        workers=args.workers,
                        project_override=spec.project_code,
                        namespace_by_site=ns_by_site,
                        on_progress=prog,
                    )
                    res.discovered_files += site_res.discovered_files
                    res.processed_files += site_res.processed_files
                    res.skipped_unchanged_files += site_res.skipped_unchanged_files
                    res.scanned_empty_files += site_res.scanned_empty_files
                    res.dropped_noisy_callouts += site_res.dropped_noisy_callouts
                    res.failed_files += site_res.failed_files
                    res.unauthorized_count += site_res.unauthorized_count
                    res.unauthorized_targets.extend(site_res.unauthorized_targets)
                    res.limitations_logged += site_res.limitations_logged
            elif args.raw_dir is not None:
                res = etl.run_local(
                    args.raw_dir, force=args.force, workers=args.workers, on_progress=prog
                )
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
                        "scanned_empty_files": res.scanned_empty_files,
                        "dropped_noisy_callouts": res.dropped_noisy_callouts,
                        "unauthorized_count": res.unauthorized_count,
                        "unauthorized_targets": res.unauthorized_targets,
                        "failed_files": res.failed_files,
                        "limitations_logged": res.limitations_logged,
                    },
                    indent=2,
                )
                + "\n"
            )
            return

        print_kv_table(
            f"ETL Summary ({format_duration(elapsed)})",
            [
                ("Discovered", f"{res.discovered_files:,}"),
                ("Processed", f"{res.processed_files:,}"),
                ("Skipped (unchanged)", f"{res.skipped_unchanged_files:,}"),
                ("Scanned / empty (needs OCR)", f"{res.scanned_empty_files:,}"),
                ("Noisy callouts dropped", f"{res.dropped_noisy_callouts:,}"),
                ("Unauthorized (401/403)", f"{res.unauthorized_count:,}"),
                ("Failed", f"{res.failed_files:,}"),
                ("Limitations logged", f"{res.limitations_logged:,}"),
            ],
        )
        if res.unauthorized_targets:
            out_console.print("[yellow]Unauthorized targets skipped:[/yellow]")
            for t in res.unauthorized_targets[:20]:
                out_console.print(f"  - {t}")
        return

    from pipeline import IndexingPipeline
    from storage import LocalHybridRepository

    with (
        LocalHybridRepository(path=str(args.qdrant_path)) as repo,
        PipelineProgress(quiet=args.quiet, json_mode=args.as_json) as prog,
    ):
        pipeline = IndexingPipeline(repository=repo, device=args.device)

        if args.etl_dir is not None:
            stats = pipeline.ingest_processed_dir(args.etl_dir, force=args.force, on_progress=prog)
        elif args.from_sharepoint:
            sites = load_sharepoint_sites(
                sites_file=args.sites_file,
                site_override=args.site,
                folder_prefix=args.folder_prefix,
                settings=settings,
            )
            stats = pipeline.ingest_sharepoint(
                sites=sites,
                catalog_path=catalog,
                processed_dir=args.processed_dir,
                force_etl=args.force,
                force_index=args.force,
                max_files=args.max_files,
                folder_prefix=args.folder_prefix,
                drop_noisy_callouts=not args.keep_noisy_callouts,
                skip_archive_folders=False if args.include_archives else None,
                workers=args.workers,
                on_progress=prog,
            )
        else:
            stats = pipeline.ingest_raw_folder(
                args.raw_dir,
                catalog_path=catalog,
                processed_dir=args.processed_dir,
                force_etl=args.force,
                force_index=args.force,
                drop_noisy_callouts=not args.keep_noisy_callouts,
                skip_archive_folders=False if args.include_archives else None,
                workers=args.workers,
                on_progress=prog,
            )

        counts = repo.collection_counts()

    elapsed = time.perf_counter() - t0
    if args.as_json:
        sys.stdout.write(
            json.dumps(
                {
                    "elapsed_seconds": round(elapsed, 2),
                    "stage_timings": stats.stage_timings,
                    "discovered_files": stats.discovered_files or stats.total_docs,
                    "total_docs": stats.total_docs,
                    "indexed_docs": stats.indexed_docs,
                    "skipped_unchanged": stats.skipped_unchanged,
                    "skipped_duplicate": stats.skipped_duplicate,
                    "skipped_archive": stats.skipped_archive,
                    "skipped_empty": stats.skipped_empty,
                    "scanned_empty_docs": stats.scanned_empty_docs,
                    "dropped_noisy_callouts": stats.dropped_noisy_callouts,
                    "unauthorized_count": stats.unauthorized_count,
                    "unauthorized_targets": stats.unauthorized_targets,
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

    summary_rows: list[tuple[str, object]] = [("Total runtime", format_duration(elapsed))]
    for stage_name, stage_secs in stats.stage_timings.items():
        label = stage_name.replace("_", " ").capitalize()
        summary_rows.append((f"  Stage · {label}", format_duration(stage_secs)))
    summary_rows.extend(
        [
            (
                "Discovered / loaded docs",
                f"{stats.discovered_files or stats.total_docs:,} / {stats.total_docs:,}",
            ),
            ("Newly indexed", f"{stats.indexed_docs:,}"),
            (
                "Skipped (unchanged / duplicate / archive)",
                f"{stats.skipped_unchanged:,} / {stats.skipped_duplicate:,} / {stats.skipped_archive:,}",
            ),
            ("Scanned / empty", f"{stats.scanned_empty_docs:,} / {stats.skipped_empty:,}"),
            ("Unauthorized (401/403)", f"{stats.unauthorized_count:,}"),
            ("Failed", f"{stats.failed_docs:,}"),
            ("Noisy callouts dropped", f"{stats.dropped_noisy_callouts:,}"),
            ("New parent / child chunks", f"{stats.total_parents:,} / {stats.total_children:,}"),
            (
                f"Qdrant totals ({args.qdrant_path})",
                f"{counts['documents']:,} docs / {counts['parent_chunks']:,} parents / {counts['child_chunks']:,} children",
            ),
        ]
    )
    print_kv_table("Ingestion Summary", summary_rows)
    if stats.unauthorized_targets:
        out_console.print("[yellow]Unauthorized targets skipped:[/yellow]")
        for t in stats.unauthorized_targets[:20]:
            out_console.print(f"  - {t}")


def main() -> None:
    run_cli(_main)


if __name__ == "__main__":
    main()
