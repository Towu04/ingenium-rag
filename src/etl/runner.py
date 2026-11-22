"""Stage 1 orchestrator: Incremental local & SharePoint extraction emitting `domain.ParsedDoc`."""

import json
import logging
import re
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from config import SUPPORTED_EXTRACTION_EXTENSIONS, get_settings
from domain import ParsedDoc, compute_doc_id, normalize_project, utc_now
from etl.catalog import DocumentRegisterIndex, ProjectCatalog, is_document_register_file
from etl.extractors import DocxExtractor, PdfExtractor, XlsxExtractor
from etl.extractors.common import render_markdown_export
from etl.metadata import MatchedMetadata, MetadataMatcher
from etl.quality import QualityLogger
from etl.sharepoint import (
    RemoteDriveItem,
    SharePointAuthError,
    SharePointConnector,
    compute_file_sha256,
)

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, int, str], None]


class ManifestEntry(BaseModel):
    """State record for a single processed file inside `manifest.json`."""

    relative_path: str
    doc_id: str
    sha256: str
    file_name: str
    project: str | None = None
    building: str | None = None
    phase: str | None = None
    doc_category: str = "unknown"
    markdown_path: str
    json_path: str
    page_count: int = 0
    table_count: int = 0
    section_count: int = 0
    extraction_confidence: float = 1.0
    is_scanned: bool = False
    is_hvac_diagram: bool = False
    needs_ocr: bool = False
    sharepoint_item_id: str | None = None
    sharepoint_etag: str | None = None
    quick_xor_hash: str | None = None
    last_modified: str | None = None
    size_bytes: int = 0
    processed_at: str = Field(default_factory=lambda: utc_now().isoformat())


class ManifestState(BaseModel):
    """Top-level structure persisted to `data/processed/manifest.json`."""

    version: str = "2.0"
    updated_at: str = Field(default_factory=lambda: utc_now().isoformat())
    files: dict[str, ManifestEntry] = Field(default_factory=dict)


class ManifestManager:
    """Manages incremental SHA-256 and SharePoint eTag/quickXorHash state in `manifest.json`."""

    def __init__(self, manifest_path: str | Path | None = None) -> None:
        self.manifest_path = (
            Path(manifest_path) if manifest_path is not None else get_settings().manifest_path
        )
        self.state: ManifestState = self._load()

    def _load(self) -> ManifestState:
        if not self.manifest_path.exists():
            return ManifestState()
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return ManifestState.model_validate(raw)
        except Exception:
            logger.warning("Could not parse manifest at %s; starting fresh.", self.manifest_path)
            return ManifestState()

    def save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.state.updated_at = utc_now().isoformat()
        self.manifest_path.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")

    def _outputs_exist(self, entry: ManifestEntry) -> bool:
        if Path(entry.markdown_path).exists() and Path(entry.json_path).exists():
            return True
        base = self.manifest_path.parent
        return (base / "markdown" / f"{entry.doc_id}.md").exists() and (
            base / "json" / f"{entry.doc_id}.json"
        ).exists()

    def is_up_to_date(self, relative_path: str, file_sha256: str) -> bool:
        entry = self.state.files.get(Path(relative_path).as_posix())
        return entry is not None and entry.sha256 == file_sha256 and self._outputs_exist(entry)

    def is_remote_up_to_date(
        self,
        relative_path: str,
        *,
        remote_etag: str | None = None,
        quick_xor_hash: str | None = None,
        sha256: str | None = None,
        last_modified: str | None = None,
        size_bytes: int | None = None,
    ) -> bool:
        entry = self.state.files.get(Path(relative_path).as_posix())
        if entry is None or not self._outputs_exist(entry):
            return False
        if remote_etag and entry.sharepoint_etag and entry.sharepoint_etag == remote_etag:
            return True
        if quick_xor_hash and entry.quick_xor_hash and entry.quick_xor_hash == quick_xor_hash:
            return True
        if sha256 and entry.sha256 == sha256:
            return True
        if (
            last_modified
            and entry.last_modified
            and entry.last_modified == last_modified
            and size_bytes is not None
            and entry.size_bytes == size_bytes
        ):
            return True
        return False

    def record_success(
        self,
        relative_path: str,
        doc: ParsedDoc,
        markdown_path: Path,
        json_path: Path,
        *,
        quick_xor_hash: str | None = None,
        last_modified: str | None = None,
        size_bytes: int = 0,
    ) -> ManifestEntry:
        norm_key = Path(relative_path).as_posix()
        meta = doc.metadata
        extra = meta.extra
        pages = {p for sec in doc.sections for p in sec.page_numbers}
        table_count = sum(1 for sec in doc.sections if sec.block_type == "table")

        entry = ManifestEntry(
            relative_path=norm_key,
            doc_id=doc.document_id,
            sha256=doc.content_hash,
            file_name=Path(meta.source_path).name,
            project=meta.project,
            building=meta.building,
            phase=meta.phase,
            doc_category=str(extra.get("doc_category", "unknown")),
            markdown_path=str(markdown_path),
            json_path=str(json_path),
            page_count=len(pages),
            table_count=table_count,
            section_count=len(doc.sections),
            extraction_confidence=float(extra.get("extraction_confidence", 1.0)),
            is_scanned=bool(extra.get("is_scanned", False)),
            is_hvac_diagram=bool(extra.get("is_hvac_diagram", False)),
            needs_ocr=bool(extra.get("needs_ocr", False)),
            sharepoint_item_id=extra.get("sharepoint_item_id"),
            sharepoint_etag=extra.get("sharepoint_etag"),
            quick_xor_hash=quick_xor_hash,
            last_modified=last_modified,
            size_bytes=size_bytes,
        )
        self.state.files[norm_key] = entry
        return entry


@dataclass
class InventoryItem:
    relative_path: str
    file_name: str
    extension: str
    size_bytes: int
    supported_for_extraction: bool
    matched_metadata: MatchedMetadata


@dataclass
class InventorySummary:
    total_files: int = 0
    extension_counts: dict[str, int] = field(default_factory=dict)
    items: list[InventoryItem] = field(default_factory=list)


@dataclass
class ETLRunResult:
    discovered_files: int = 0
    processed_files: int = 0
    skipped_unchanged_files: int = 0
    skipped_archive_files: int = 0
    scanned_empty_files: int = 0
    dropped_noisy_callouts: int = 0
    failed_files: int = 0
    unauthorized_count: int = 0
    unauthorized_targets: list[str] = field(default_factory=list)
    limitations_logged: int = 0
    documents: list[ParsedDoc] = field(default_factory=list)
    output_json_paths: list[Path] = field(default_factory=list)


def is_archive_path(relative_path: str | Path, archive_names: Iterable[str]) -> bool:
    """Return True if any parent folder segment matches a configured archive/draft/rejected name."""
    norm_names = [n.strip().lower() for n in archive_names if n.strip()]
    if not norm_names:
        return False
    patterns = [re.compile(rf"(?<!\w){re.escape(n)}(?!\w)") for n in norm_names]
    for part in Path(relative_path).parent.parts:
        part_low = part.strip().lower()
        if any(p.search(part_low) for p in patterns):
            return True
    return False


class ETLRunner:
    """Incremental document extraction & metadata matching emitting canonical `ParsedDoc`s."""

    def __init__(
        self,
        catalog_path: str | Path | None = None,
        markdown_out_dir: str | Path | None = None,
        json_out_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        limitations_report_path: str | Path | None = None,
        drop_noisy_callouts: bool = True,
        skip_archive_folders: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.markdown_out_dir = (
            Path(markdown_out_dir) if markdown_out_dir is not None else settings.markdown_out_dir
        )
        self.json_out_dir = (
            Path(json_out_dir) if json_out_dir is not None else settings.json_out_dir
        )
        self.manifest = ManifestManager(manifest_path=manifest_path or settings.manifest_path)
        self.quality_logger = QualityLogger(
            report_path=limitations_report_path or settings.limitations_report_path
        )
        self.drop_noisy_callouts = drop_noisy_callouts
        self.skip_archive_folders = (
            settings.skip_archive_folders if skip_archive_folders is None else skip_archive_folders
        )
        self.archive_folder_names = settings.archive_folder_names

        catalog = (
            ProjectCatalog.from_file(catalog_path)
            if catalog_path and Path(catalog_path).exists()
            else ProjectCatalog()
        )
        self.register_index = DocumentRegisterIndex()
        self.metadata_matcher = MetadataMatcher(catalog=catalog, register_index=self.register_index)

        self.pdf_extractor = PdfExtractor(metadata_matcher=self.metadata_matcher)
        self.docx_extractor = DocxExtractor(metadata_matcher=self.metadata_matcher)
        self.xlsx_extractor = XlsxExtractor(metadata_matcher=self.metadata_matcher)
        self._state_lock = threading.Lock()
        self._pdf_lock = threading.Lock()
        self._unsaved_manifest_updates = 0

    def _append_inventory_item(
        self,
        summary: InventorySummary,
        *,
        rel_path: Path,
        file_name: str,
        size_bytes: int,
        sharepoint_url: str | None = None,
    ) -> None:
        ext = rel_path.suffix.lower() or "(no_ext)"
        summary.total_files += 1
        summary.extension_counts[ext] = summary.extension_counts.get(ext, 0) + 1
        summary.items.append(
            InventoryItem(
                relative_path=rel_path.as_posix(),
                file_name=file_name,
                extension=ext,
                size_bytes=size_bytes,
                supported_for_extraction=ext in SUPPORTED_EXTRACTION_EXTENSIONS,
                matched_metadata=self.metadata_matcher.match(
                    rel_path, sharepoint_url=sharepoint_url
                ),
            )
        )

    def _persist_extracted_doc(
        self,
        parsed_doc: ParsedDoc,
        rel_posix: str,
        result: ETLRunResult,
        *,
        json_path: Path | None = None,
        quick_xor_hash: str | None = None,
        last_modified: str | None = None,
        size_bytes: int = 0,
        save_every: int = 20,
    ) -> Path:
        md_path, out_json = self._write_outputs(parsed_doc, json_path=json_path)
        with self._state_lock:
            if self.quality_logger.evaluate_and_log(parsed_doc) is not None:
                result.limitations_logged += 1
            result.dropped_noisy_callouts += int(
                parsed_doc.metadata.extra.get("dropped_noisy_callouts") or 0
            )
            if not parsed_doc.sections and (
                parsed_doc.metadata.extra.get("is_scanned")
                or parsed_doc.metadata.extra.get("needs_ocr")
            ):
                result.scanned_empty_files += 1
            self.manifest.record_success(
                rel_posix,
                parsed_doc,
                md_path,
                out_json,
                quick_xor_hash=quick_xor_hash,
                last_modified=last_modified,
                size_bytes=size_bytes,
            )
            self._unsaved_manifest_updates += 1
            if self._unsaved_manifest_updates >= save_every:
                self.manifest.save()
                self._unsaved_manifest_updates = 0
            result.processed_files += 1
            result.output_json_paths.append(out_json)
            result.documents.append(parsed_doc)
        return out_json

    def _flush_manifest(self) -> None:
        with self._state_lock:
            if self._unsaved_manifest_updates > 0:
                self.manifest.save()
                self._unsaved_manifest_updates = 0

    def _record_unauthorized(
        self,
        target: str,
        exc: SharePointAuthError,
        result: ETLRunResult,
        *,
        project: str | None = None,
    ) -> None:
        with self._state_lock:
            result.unauthorized_count += 1
            result.unauthorized_targets.append(f"HTTP {exc.status_code}: {target}")
            self.quality_logger.log_unauthorized(
                target, exc.status_code, exc.reason, project=project
            )
            result.limitations_logged += 1

    def _record_failure(
        self,
        rel_path: Path,
        exc: Exception,
        result: ETLRunResult,
        *,
        sharepoint_url: str | None = None,
    ) -> None:
        matched = self.metadata_matcher.match(rel_path, sharepoint_url=sharepoint_url)
        with self._state_lock:
            result.failed_files += 1
            self.quality_logger.log_corrupted_file(
                rel_path,
                str(exc),
                project=matched.project,
                building=matched.building,
                phase=matched.phase,
            )
            result.limitations_logged += 1

    def scan_inventory(self, raw_dir: str | Path) -> InventorySummary:
        """Scan a local folder tree and preview resolved metadata without running OCR."""
        root = Path(raw_dir)
        if not root.exists():
            raise FileNotFoundError(f"Raw directory not found: {root}")
        self.register_index.discover_and_load(root)

        summary = InventorySummary()
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            self._append_inventory_item(
                summary,
                rel_path=file_path.relative_to(root),
                file_name=file_path.name,
                size_bytes=file_path.stat().st_size,
            )
        return summary

    def scan_sharepoint_inventory(
        self,
        connector: SharePointConnector | None = None,
        *,
        folder_prefix: str | None = None,
    ) -> InventorySummary:
        """List remote SharePoint files and preview metadata without downloading binaries."""
        sp = connector or SharePointConnector.from_env(json_out_dir=self.json_out_dir)
        remote_items = sp.list_remote_files(supported_only=False, folder_prefix=folder_prefix)

        for item in remote_items:
            if is_document_register_file(item.relative_path):
                local_reg: Path | None = None
                try:
                    local_reg = sp.download_file(item)
                    self.register_index.load_register_file(local_reg)
                except Exception:
                    logger.warning("Failed to load remote register '%s'", item.relative_path)
                finally:
                    if local_reg is not None:
                        sp.cleanup_local_file(local_reg)

        summary = InventorySummary()
        for item in remote_items:
            self._append_inventory_item(
                summary,
                rel_path=Path(item.relative_path),
                file_name=item.name,
                size_bytes=item.size_bytes,
                sharepoint_url=item.web_url,
            )
        return summary

    def run_local(
        self,
        raw_dir: str | Path,
        *,
        force: bool = False,
        workers: int = 1,
        on_progress: ProgressCallback | None = None,
    ) -> ETLRunResult:
        """Incrementally extract all supported files in `raw_dir` into `ParsedDoc` JSON/MD."""
        root = Path(raw_dir)
        if not root.exists():
            raise FileNotFoundError(f"Raw directory not found: {root}")
        self.markdown_out_dir.mkdir(parents=True, exist_ok=True)
        self.json_out_dir.mkdir(parents=True, exist_ok=True)

        loaded_regs = self.register_index.discover_and_load(root)
        logger.info("Pass 1: indexed %d Document Register entries under %s.", loaded_regs, root)

        all_supported = [
            p
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in SUPPORTED_EXTRACTION_EXTENSIONS
        ]
        result = ETLRunResult()
        if self.skip_archive_folders:
            supported_files = []
            for p in all_supported:
                if is_archive_path(p.relative_to(root), self.archive_folder_names):
                    result.skipped_archive_files += 1
                else:
                    supported_files.append(p)
        else:
            supported_files = all_supported
        total = len(supported_files)
        done = 0
        logger.info("Pass 2: processing %d supported local file(s)...", total)
        if on_progress and total:
            on_progress("bar_start", 0, total, "ETL Extraction")

        def _handle_local(idx: int, file_path: Path) -> str:
            rel_path = file_path.relative_to(root)
            rel_posix = rel_path.as_posix()
            file_hash = compute_file_sha256(file_path)

            with self._state_lock:
                result.discovered_files += 1
                is_up = not force and self.manifest.is_up_to_date(rel_posix, file_hash)
                if is_up:
                    result.skipped_unchanged_files += 1
            if is_up:
                logger.debug("[%d/%d] SKIP unchanged: %s", idx, total, rel_posix)
                return f"skip {file_path.name}"

            try:
                logger.debug("[%d/%d] Extracting: %s", idx, total, rel_posix)
                parsed_doc = self._dispatch(file_path, rel_path)
                parsed_doc.content_hash = file_hash
                json_path = self._persist_extracted_doc(
                    parsed_doc, rel_posix, result, size_bytes=file_path.stat().st_size
                )
                logger.debug(
                    "[%d/%d] DONE: %s -> %s (%d sections)",
                    idx,
                    total,
                    file_path.name,
                    json_path.name,
                    len(parsed_doc.sections),
                )
            except Exception as exc:
                logger.error("[%d/%d] ERROR extracting %s: %s", idx, total, rel_posix, exc)
                self._record_failure(rel_path, exc, result)
            return file_path.name

        try:
            eff_workers = max(1, workers)
            if eff_workers == 1 or total <= 1:
                for idx, fp in enumerate(supported_files, start=1):
                    label = _handle_local(idx, fp)
                    done += 1
                    if on_progress:
                        on_progress("bar_advance", done, total, label)
            else:
                with ThreadPoolExecutor(max_workers=eff_workers) as pool:
                    futures = [
                        pool.submit(_handle_local, idx, fp)
                        for idx, fp in enumerate(supported_files, start=1)
                    ]
                    for fut in as_completed(futures):
                        label = fut.result()
                        done += 1
                        if on_progress:
                            on_progress("bar_advance", done, total, label)
        finally:
            self._flush_manifest()

        if on_progress and total:
            on_progress("bar_done", total, total, "ETL Extraction")
        logger.info(
            "Local ETL complete: processed=%d skipped=%d failed=%d limitations=%d",
            result.processed_files,
            result.skipped_unchanged_files,
            result.failed_files,
            result.limitations_logged,
        )
        return result

    def run_sharepoint(
        self,
        connector: SharePointConnector | None = None,
        *,
        force: bool = False,
        max_files: int | None = None,
        folder_prefix: str | None = None,
        extensions: set[str] | None = None,
        workers: int = 1,
        project_override: str | None = None,
        namespace_by_site: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> ETLRunResult:
        """Stream files from SharePoint with bounded concurrency, skipping unchanged files pre-download."""
        sp = connector or SharePointConnector.from_env(json_out_dir=self.json_out_dir)
        self.markdown_out_dir.mkdir(parents=True, exist_ok=True)
        self.json_out_dir.mkdir(parents=True, exist_ok=True)

        result = ETLRunResult()
        site_label = f"{sp.config.hostname}{sp.config.site_path}"
        norm_proj_override = normalize_project(project_override) if project_override else None

        try:
            remote_items = sp.list_remote_files(supported_only=True, folder_prefix=folder_prefix)
        except SharePointAuthError as auth_err:
            logger.warning("Skipping unauthorized SharePoint site '%s': %s", site_label, auth_err)
            self._record_unauthorized(site_label, auth_err, result, project=norm_proj_override)
            return result

        if extensions:
            norm_exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
            remote_items = [item for item in remote_items if item.extension in norm_exts]

        if self.skip_archive_folders:
            active_items = []
            for item in remote_items:
                if is_archive_path(item.relative_path, self.archive_folder_names):
                    result.skipped_archive_files += 1
                else:
                    active_items.append(item)
            if result.skipped_archive_files:
                logger.info(
                    "Skipped %d archive/draft file(s) in %s.",
                    result.skipped_archive_files,
                    sp.config.site_slug,
                )
            remote_items = active_items

        register_items = [
            item for item in remote_items if is_document_register_file(item.relative_path)
        ]
        regular_items = [
            item for item in remote_items if not is_document_register_file(item.relative_path)
        ]

        if max_files is not None:
            reg_budget = min(len(register_items), 1) if max_files >= 2 else 0
            register_items = register_items[:reg_budget]
            regular_items = regular_items[: max(0, max_files - len(register_items))]

        logger.info(
            "Starting 2-pass SharePoint extraction (%s): %d register(s) + %d regular file(s).",
            sp.config.site_slug,
            len(register_items),
            len(regular_items),
        )
        total_items = len(register_items) + len(regular_items)
        done_items = 0
        bar_title = f"SharePoint ({sp.config.site_slug})"
        if on_progress and total_items:
            on_progress("bar_start", 0, total_items, bar_title)

        try:
            # Pass 1: Document Registers (sequential so register index is populated before Pass 2)
            for idx, item in enumerate(register_items, start=1):
                self._process_remote_item(
                    sp,
                    item,
                    idx,
                    len(register_items),
                    is_register=True,
                    force=force,
                    result=result,
                    project_override=norm_proj_override,
                    namespace_by_site=namespace_by_site,
                )
                done_items += 1
                if on_progress:
                    on_progress("bar_advance", done_items, total_items, item.name)

            # Pass 2: Regular files (bounded concurrency for I/O + extraction)
            eff_workers = max(1, workers)
            if eff_workers == 1 or len(regular_items) <= 1:
                for idx, item in enumerate(regular_items, start=1):
                    self._process_remote_item(
                        sp,
                        item,
                        idx,
                        len(regular_items),
                        is_register=False,
                        force=force,
                        result=result,
                        project_override=norm_proj_override,
                        namespace_by_site=namespace_by_site,
                    )
                    done_items += 1
                    if on_progress:
                        on_progress("bar_advance", done_items, total_items, item.name)
            else:
                with ThreadPoolExecutor(max_workers=eff_workers) as pool:
                    futures = {
                        pool.submit(
                            self._process_remote_item,
                            sp,
                            item,
                            idx,
                            len(regular_items),
                            is_register=False,
                            force=force,
                            result=result,
                            project_override=norm_proj_override,
                            namespace_by_site=namespace_by_site,
                        ): item
                        for idx, item in enumerate(regular_items, start=1)
                    }
                    for fut in as_completed(futures):
                        item = futures[fut]
                        fut.result()
                        done_items += 1
                        if on_progress:
                            on_progress("bar_advance", done_items, total_items, item.name)
        finally:
            self._flush_manifest()

        if on_progress and total_items:
            on_progress("bar_done", total_items, total_items, bar_title)
        logger.info(
            "SharePoint ETL complete (%s): processed=%d skipped=%d unauthorized=%d failed=%d",
            sp.config.site_slug,
            result.processed_files,
            result.skipped_unchanged_files,
            result.unauthorized_count,
            result.failed_files,
        )
        return result

    def _process_remote_item(
        self,
        sp: SharePointConnector,
        item: RemoteDriveItem,
        idx: int,
        total: int,
        *,
        is_register: bool,
        force: bool,
        result: ETLRunResult,
        project_override: str | None = None,
        namespace_by_site: bool = False,
    ) -> None:
        eff_rel_posix = (
            f"{sp.config.site_slug}/{Path(item.relative_path).as_posix()}"
            if namespace_by_site
            else Path(item.relative_path).as_posix()
        )
        rel_path = Path(eff_rel_posix)
        pass_label = "Pass 1" if is_register else "Pass 2"
        local_path: Path | None = None

        with self._state_lock:
            result.discovered_files += 1
            up_to_date = not force and self.manifest.is_remote_up_to_date(
                eff_rel_posix,
                remote_etag=item.etag or item.ctag,
                quick_xor_hash=item.quick_xor_hash,
                sha256=item.sha256_hash,
                last_modified=item.last_modified,
                size_bytes=item.size_bytes,
            )
            if up_to_date:
                existing_entry = self.manifest.state.files.get(eff_rel_posix)
                if existing_entry is not None:
                    existing_json = Path(existing_entry.json_path)
                    if not existing_json.exists():
                        existing_json = self.json_out_dir / f"{existing_entry.doc_id}.json"
                    if existing_json.exists():
                        result.output_json_paths.append(existing_json)
                if not is_register:
                    result.skipped_unchanged_files += 1

        try:
            if up_to_date:
                if is_register:
                    local_path = sp.download_file(item, target_rel_path=eff_rel_posix)
                    self.register_index.load_register_file(local_path)
                    with self._state_lock:
                        result.skipped_unchanged_files += 1
                logger.debug("[%s: %d/%d] SKIP unchanged: %s", pass_label, idx, total, item.name)
                return

            logger.debug(
                "[%s: %d/%d] Streaming & extracting: %s", pass_label, idx, total, eff_rel_posix
            )
            local_path = sp.download_file(item, target_rel_path=eff_rel_posix)
            file_hash = compute_file_sha256(local_path)
            doc_id = compute_doc_id(eff_rel_posix)
            if is_register:
                self.register_index.load_register_file(local_path)

            parsed_doc = self._dispatch(
                local_path,
                rel_path,
                sharepoint_url=item.web_url,
            )
            parsed_doc.document_id = doc_id
            parsed_doc.content_hash = file_hash
            if project_override:
                parsed_doc.metadata.project = project_override
            parsed_doc.metadata.extra["sharepoint_item_id"] = item.item_id
            parsed_doc.metadata.extra["sharepoint_etag"] = item.etag or item.ctag
            if item.web_url:
                parsed_doc.metadata.sharepoint_url = item.web_url

            json_path = self._persist_extracted_doc(
                parsed_doc,
                eff_rel_posix,
                result,
                quick_xor_hash=item.quick_xor_hash,
                last_modified=item.last_modified,
                size_bytes=item.size_bytes,
            )
            logger.debug(
                "[%s: %d/%d] DONE: %s -> %s", pass_label, idx, total, item.name, json_path.name
            )

        except SharePointAuthError as auth_err:
            logger.warning(
                "[%s: %d/%d] UNAUTHORIZED (%d) on %s",
                pass_label,
                idx,
                total,
                auth_err.status_code,
                eff_rel_posix,
            )
            self._record_unauthorized(eff_rel_posix, auth_err, result, project=project_override)
        except Exception as exc:
            logger.error("[%s: %d/%d] ERROR on %s: %s", pass_label, idx, total, item.name, exc)
            self._record_failure(rel_path, exc, result, sharepoint_url=item.web_url)
        finally:
            if local_path is not None:
                sp.cleanup_local_file(local_path)

    def _dispatch(
        self,
        file_path: Path,
        rel_path: Path,
        *,
        sharepoint_url: str | None = None,
    ) -> ParsedDoc:
        ext = file_path.suffix.lower()
        kwargs = {
            "relative_path": rel_path,
            "sharepoint_url": sharepoint_url,
            "drop_noisy_callouts": self.drop_noisy_callouts,
        }
        if ext == ".pdf":
            with self._pdf_lock:
                return self.pdf_extractor.extract(file_path, **kwargs)
        if ext == ".docx":
            return self.docx_extractor.extract(file_path, **kwargs)
        if ext == ".xlsx":
            return self.xlsx_extractor.extract(file_path, **kwargs)
        raise ValueError(f"Unsupported file extension for extraction: {ext}")

    def _write_outputs(
        self,
        doc: ParsedDoc,
        *,
        json_path: Path | None = None,
        md_path: Path | None = None,
    ) -> tuple[Path, Path]:
        target_md = md_path or (self.markdown_out_dir / f"{doc.document_id}.md")
        target_json = json_path or (self.json_out_dir / f"{doc.document_id}.json")
        target_md.parent.mkdir(parents=True, exist_ok=True)
        target_json.parent.mkdir(parents=True, exist_ok=True)
        target_md.write_text(render_markdown_export(doc), encoding="utf-8")
        target_json.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        return target_md, target_json

    def get_summary_report(self) -> dict[str, Any]:
        entries = list(self.manifest.state.files.values())
        limitations = self.quality_logger.read_records()
        by_category: dict[str, int] = {}
        for e in entries:
            by_category[e.doc_category] = by_category.get(e.doc_category, 0) + 1
        return {
            "total_documents": len(entries),
            "total_pages": sum(e.page_count for e in entries),
            "total_tables": sum(e.table_count for e in entries),
            "total_sections": sum(e.section_count for e in entries),
            "needs_ocr_count": sum(1 for e in entries if e.needs_ocr),
            "by_category": by_category,
            "entries": entries,
            "limitations": limitations,
        }


def load_processed_docs(
    processed_dir: str | Path,
    *,
    use_manifest: bool = True,
    dedupe_by_hash: bool = True,
) -> list[ParsedDoc]:
    """Load clean `domain.ParsedDoc` JSON files from `processed_dir` (`data/processed/`)."""
    base = Path(processed_dir)
    json_dir = base / "json"
    manifest_path = base / "manifest.json"

    json_paths: list[Path] = []
    if use_manifest and manifest_path.exists():
        manifest = ManifestState.model_validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        for entry in manifest.files.values():
            portable = json_dir / f"{entry.doc_id}.json"
            if portable.exists():
                json_paths.append(portable)
            elif Path(entry.json_path).exists():
                json_paths.append(Path(entry.json_path))
            else:
                logger.warning("Manifest entry '%s' missing JSON at %s", entry.file_name, portable)
    elif json_dir.exists():
        json_paths = sorted(json_dir.glob("*.json"))
    else:
        raise FileNotFoundError(f"Processed JSON directory not found: {json_dir}")

    docs: list[ParsedDoc] = []
    seen_by_hash: dict[str, ParsedDoc] = {}
    for jp in json_paths:
        parsed = ParsedDoc.model_validate_json(jp.read_text(encoding="utf-8"))
        if not parsed.sections and not parsed.markdown.strip():
            if parsed.metadata.extra.get("is_scanned") or parsed.metadata.extra.get("needs_ocr"):
                logger.warning(
                    "Skipping scanned PDF with 0 sections (needs_ocr=True): %s",
                    parsed.metadata.source_path,
                )
            else:
                logger.warning("Skipping empty/orphan JSON with 0 sections: %s", jp.name)
            continue

        if dedupe_by_hash and parsed.content_hash in seen_by_hash:
            canonical = seen_by_hash[parsed.content_hash]
            canonical.metadata.extra.setdefault("duplicate_source_paths", []).append(
                parsed.metadata.source_path
            )
            logger.info(
                "Deduplicated byte-identical file '%s' (same SHA-256 as '%s').",
                parsed.metadata.source_path,
                canonical.metadata.source_path,
            )
            continue

        seen_by_hash[parsed.content_hash] = parsed
        docs.append(parsed)

    logger.info("Loaded %d ParsedDoc(s) from %s.", len(docs), base)
    return docs
