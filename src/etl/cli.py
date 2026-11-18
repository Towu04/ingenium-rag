"""Typer CLI entry point for the EMERGENT x Ingenium ETL & Metadata Matching pipeline.

Commands:
  1. `python -m etl.cli inventory <raw_dir>` or `python -m etl.cli inventory --from-sharepoint`
  2. `python -m etl.cli run <raw_dir> --catalog <catalog.csv>` or `python -m etl.cli run --from-sharepoint`
  3. `python -m etl.cli report`
"""

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
import typer

from etl.config import (
    DEFAULT_CATALOG_PATH,
    JSON_OUT_DIR,
    LIMITATIONS_REPORT_PATH,
    MANIFEST_PATH,
    MARKDOWN_OUT_DIR,
    RAW_SHAREPOINT_DIR,
)
from etl.pipeline import ETLPipeline
from etl.sharepoint_access import SharePointConnector

app = typer.Typer(
    name="ingenium-etl",
    help="EMERGENT x Ingenium — Local ETL, Metadata Matching & Document Extraction CLI",
    add_completion=False,
)
console = Console()


@app.command("inventory")
def inventory_cmd(
    raw_dir: Optional[Path] = typer.Argument(
        None,
        help="Path to local raw SharePoint project folders (omit when using --from-sharepoint).",
    ),
    from_sharepoint: bool = typer.Option(
        False,
        "--from-sharepoint",
        help="List remote SharePoint files via Microsoft Graph API without downloading heavy binaries.",
    ),
    catalog: Optional[Path] = typer.Option(
        None,
        "--catalog",
        "-c",
        help="Optional Google Sheet project catalog (.csv or .xlsx).",
    ),
) -> None:
    """Scan a local or remote SharePoint folder, count file extensions, and preview 3-layer metadata."""
    resolved_catalog = catalog if (catalog and catalog.exists()) else (
        DEFAULT_CATALOG_PATH if DEFAULT_CATALOG_PATH.exists() else None
    )
    pipeline = ETLPipeline(catalog_path=resolved_catalog)

    if from_sharepoint:
        connector = SharePointConnector.from_env(staging_dir=raw_dir or RAW_SHAREPOINT_DIR)
        summary = pipeline.scan_sharepoint_inventory(connector=connector)
        title_source = f"SharePoint Graph ({connector.config.hostname}{connector.config.site_path})"
    else:
        if raw_dir is None:
            raise typer.BadParameter("Provide a local <raw_dir> path or pass --from-sharepoint.")
        if not raw_dir.exists() or not raw_dir.is_dir():
            raise typer.BadParameter(f"Directory does not exist: {raw_dir}")
        summary = pipeline.scan_inventory(raw_dir)
        title_source = str(raw_dir)

    ext_table = Table(title=f"SharePoint Inventory: {title_source}")
    ext_table.add_column("Extension", style="cyan")
    ext_table.add_column("Count", justify="right", style="green")
    for ext, count in sorted(summary.extension_counts.items()):
        ext_table.add_row(ext, str(count))
    ext_table.add_row("TOTAL", str(summary.total_files), style="bold")
    console.print(ext_table)

    meta_table = Table(title="Inferred Domain Metadata Preview")
    meta_table.add_column("Relative Path", style="white")
    meta_table.add_column("Project ID", style="cyan")
    meta_table.add_column("Building / Zone", style="magenta")
    meta_table.add_column("Phase", style="yellow")
    meta_table.add_column("Category", style="green")

    for item in summary.items:
        m = item.matched_metadata
        meta_table.add_row(
            item.relative_path,
            m.project_id,
            m.building,
            m.phase,
            m.doc_category.value,
        )
    console.print(meta_table)


@app.command("run")
def run_cmd(
    raw_dir: Optional[Path] = typer.Argument(
        None,
        help="Path to local raw SharePoint project folders (omit when using --from-sharepoint).",
    ),
    from_sharepoint: bool = typer.Option(
        False,
        "--from-sharepoint",
        help="Stream files from SharePoint one by one, enrich JSON in-place, and delete raw files.",
    ),
    catalog: Optional[Path] = typer.Option(
        None,
        "--catalog",
        "-c",
        help="Path to exported Google Sheet project catalog (.csv or .xlsx).",
    ),
    markdown_out: Path = typer.Option(
        MARKDOWN_OUT_DIR,
        "--markdown-out",
        help="Output directory for anchored .md files.",
    ),
    json_out: Path = typer.Option(
        JSON_OUT_DIR,
        "--json-out",
        help="Output directory for canonical .json files.",
    ),
    manifest: Path = typer.Option(
        MANIFEST_PATH,
        "--manifest",
        help="Path to incremental SHA-256 state manifest.json.",
    ),
    limitations_log: Path = typer.Option(
        LIMITATIONS_REPORT_PATH,
        "--limitations-log",
        help="Path to limitations_report.jsonl.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-extract all files even if unchanged in manifest.json.",
    ),
) -> None:
    """Run incremental document extraction (from local directory or ephemeral SharePoint stream)."""
    resolved_catalog = catalog if (catalog and catalog.exists()) else (
        DEFAULT_CATALOG_PATH if DEFAULT_CATALOG_PATH.exists() else None
    )
    pipeline = ETLPipeline(
        catalog_path=resolved_catalog,
        markdown_out_dir=markdown_out,
        json_out_dir=json_out,
        manifest_path=manifest,
        limitations_report_path=limitations_log,
    )

    if from_sharepoint:
        connector = SharePointConnector.from_env(
            staging_dir=raw_dir or RAW_SHAREPOINT_DIR,
            json_out_dir=json_out,
        )
        result = pipeline.run_from_sharepoint(connector=connector, force=force)
    else:
        if raw_dir is None:
            raise typer.BadParameter("Provide a local <raw_dir> path or pass --from-sharepoint.")
        if not raw_dir.exists() or not raw_dir.is_dir():
            raise typer.BadParameter(f"Directory does not exist: {raw_dir}")
        result = pipeline.run(raw_dir, force=force)

    summary_table = Table(title="Incremental ETL Run Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", justify="right", style="green")
    summary_table.add_row("Discovered Supported Files", str(result.discovered_files))
    summary_table.add_row("Newly Processed Files", str(result.processed_files))
    summary_table.add_row("Skipped Unchanged Files (SHA-256)", str(result.skipped_unchanged_files))
    summary_table.add_row("Failed / Corrupted Files", str(result.failed_files))
    summary_table.add_row("Limitations Logged", str(result.limitations_logged))
    console.print(summary_table)


@app.command("report")
def report_cmd(
    manifest: Path = typer.Option(
        MANIFEST_PATH,
        "--manifest",
        help="Path to manifest.json.",
    ),
    limitations_log: Path = typer.Option(
        LIMITATIONS_REPORT_PATH,
        "--limitations-log",
        help="Path to limitations_report.jsonl.",
    ),
) -> None:
    """Print summary table of extracted documents, page/table counts, and logged limitations."""
    pipeline = ETLPipeline(
        manifest_path=manifest,
        limitations_report_path=limitations_log,
    )
    report_data = pipeline.get_summary_report()

    doc_table = Table(title="Extracted Documents Report (AZ Sint-Jan Brugge)")
    doc_table.add_column("Doc ID", style="cyan")
    doc_table.add_column("File Name", style="white")
    doc_table.add_column("Project", style="green")
    doc_table.add_column("Building", style="magenta")
    doc_table.add_column("Phase", style="yellow")
    doc_table.add_column("Category", style="blue")
    doc_table.add_column("Pages", justify="right")
    doc_table.add_column("Tables", justify="right")
    doc_table.add_column("Confidence", justify="right")

    for entry in report_data["entries"]:
        doc_table.add_row(
            entry.doc_id,
            entry.file_name,
            entry.project_id,
            entry.building,
            entry.phase,
            entry.doc_category,
            str(entry.page_count),
            str(entry.table_count),
            f"{entry.extraction_confidence:.2f}",
        )
    console.print(doc_table)

    console.print(
        f"[bold]Totals:[/bold] {report_data['total_documents']} documents | "
        f"{report_data['total_pages']} PDF pages | "
        f"{report_data['total_tables']} tables | "
        f"{len(report_data['limitations'])} logged limitations"
    )

    if report_data["limitations"]:
        lim_table = Table(title="Logged Extraction Limitations (Deliverable 3 Audit)")
        lim_table.add_column("File Name", style="red")
        lim_table.add_column("Limitation Type", style="yellow")
        lim_table.add_column("Confidence", justify="right")
        lim_table.add_column("Recommended Action", style="white")
        for rec in report_data["limitations"]:
            lim_table.add_row(
                rec.file_name,
                rec.limitation_type,
                f"{rec.extraction_confidence:.2f}",
                rec.recommended_action,
            )
        console.print(lim_table)


if __name__ == "__main__":
    app()
