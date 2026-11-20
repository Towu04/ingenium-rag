"""Stage 1 orchestrator: Incremental local & SharePoint extraction emitting `domain.ParsedDoc`."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from config import SUPPORTED_EXTRACTION_EXTENSIONS, get_settings
from domain import ParsedDoc, utc_now
from etl.catalog import DocumentRegisterIndex, ProjectCatalog, is_document_register_file
from etl.extractors import DocxExtractor, PdfExtractor, XlsxExtractor
from etl.extractors.common import render_markdown_export
from etl.metadata import MatchedMetadata, MetadataMatcher
from etl.quality import QualityLogger
from etl.sharepoint import RemoteDriveItem, SharePointConnector, compute_file_sha256

logger = logging.getLogger(__name__)


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
    ) -> bool:
        entry = self.state.files.get(Path(relative_path).as_posix())
        if entry is None or not self._outputs_exist(entry):
            return False
        if remote_etag and entry.sharepoint_etag and entry.sharepoint_etag == remote_etag:
            return True
        if quick_xor_hash and entry.quick_xor_hash and entry.quick_xor_hash == quick_xor_hash:
            return True
        return bool(sha256 and entry.sha256 == sha256)

    def record_success(
        self,
        relative_path: str,
        doc: ParsedDoc,
        markdown_path: Path,
        json_path: Path,
        *,
        quick_xor_hash: str | None = None,
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
    failed_files: int = 0
    limitations_logged: int = 0
    documents: list[ParsedDoc] = field(default_factory=list)


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
            rel_path = file_path.relative_to(root)
            ext = file_path.suffix.lower() or "(no_ext)"
            summary.total_files += 1
            summary.extension_counts[ext] = summary.extension_counts.get(ext, 0) + 1
            summary.items.append(
                InventoryItem(
                    relative_path=rel_path.as_posix(),
                    file_name=file_path.name,
                    extension=ext,
                    size_bytes=file_path.stat().st_size,
                    supported_for_extraction=ext in SUPPORTED_EXTRACTION_EXTENSIONS,
                    matched_metadata=self.metadata_matcher.match(rel_path),
                )
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
            ext = item.extension or "(no_ext)"
            summary.total_files += 1
            summary.extension_counts[ext] = summary.extension_counts.get(ext, 0) + 1
            matched = self.metadata_matcher.match(item.relative_path, sharepoint_url=item.web_url)
            summary.items.append(
                InventoryItem(
                    relative_path=Path(item.relative_path).as_posix(),
                    file_name=item.name,
                    extension=ext,
                    size_bytes=item.size_bytes,
                    supported_for_extraction=ext in SUPPORTED_EXTRACTION_EXTENSIONS,
                    matched_metadata=matched,
                )
            )
        return summary

    def run_local(self, raw_dir: str | Path, *, force: bool = False) -> ETLRunResult:
        """Incrementally extract all supported files in `raw_dir` into `ParsedDoc` JSON/MD."""
        root = Path(raw_dir)
        if not root.exists():
            raise FileNotFoundError(f"Raw directory not found: {root}")
        self.markdown_out_dir.mkdir(parents=True, exist_ok=True)
        self.json_out_dir.mkdir(parents=True, exist_ok=True)

        loaded_regs = self.register_index.discover_and_load(root)
        logger.info("Pass 1: indexed %d Document Register entries under %s.", loaded_regs, root)

        supported_files = [
            p
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in SUPPORTED_EXTRACTION_EXTENSIONS
        ]
        result = ETLRunResult()
        total = len(supported_files)
        logger.info("Pass 2: processing %d supported local file(s)...", total)

        for idx, file_path in enumerate(supported_files, start=1):
            result.discovered_files += 1
            rel_path = file_path.relative_to(root)
            rel_posix = rel_path.as_posix()
            file_hash = compute_file_sha256(file_path)

            if not force and self.manifest.is_up_to_date(rel_posix, file_hash):
                result.skipped_unchanged_files += 1
                logger.debug("[%d/%d] SKIP unchanged: %s", idx, total, rel_posix)
                continue

            try:
                logger.debug("[%d/%d] Extracting: %s", idx, total, rel_posix)
                parsed_doc = self._dispatch(file_path, rel_path)
                parsed_doc.content_hash = file_hash

                if self.quality_logger.evaluate_and_log(parsed_doc) is not None:
                    result.limitations_logged += 1

                md_path, json_path = self._write_outputs(parsed_doc)
                self.manifest.record_success(rel_posix, parsed_doc, md_path, json_path)
                self.manifest.save()

                result.processed_files += 1
                result.documents.append(parsed_doc)
                logger.debug(
                    "[%d/%d] DONE: %s -> %s (%d sections)",
                    idx,
                    total,
                    file_path.name,
                    json_path.name,
                    len(parsed_doc.sections),
                )
            except Exception as exc:
                result.failed_files += 1
                logger.error("[%d/%d] ERROR extracting %s: %s", idx, total, rel_posix, exc)
                matched = self.metadata_matcher.match(rel_path)
                self.quality_logger.log_corrupted_file(
                    rel_path,
                    str(exc),
                    project=matched.project,
                    building=matched.building,
                    phase=matched.phase,
                )
                result.limitations_logged += 1

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
    ) -> ETLRunResult:
        """Stream files from SharePoint one-by-one, enrich `ParsedDoc` JSON in-place, and delete binaries."""
        sp = connector or SharePointConnector.from_env(json_out_dir=self.json_out_dir)
        self.markdown_out_dir.mkdir(parents=True, exist_ok=True)
        self.json_out_dir.mkdir(parents=True, exist_ok=True)

        remote_items = sp.list_remote_files(supported_only=True, folder_prefix=folder_prefix)
        if extensions:
            norm_exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
            remote_items = [item for item in remote_items if item.extension in norm_exts]

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
            "Starting 2-pass SharePoint extraction: %d register(s) + %d regular file(s).",
            len(register_items),
            len(regular_items),
        )
        result = ETLRunResult()

        # Pass 1: Document Registers
        for idx, item in enumerate(register_items, start=1):
            self._process_remote_item(
                sp, item, idx, len(register_items), is_register=True, force=force, result=result
            )

        # Pass 2: Regular files
        for idx, item in enumerate(regular_items, start=1):
            self._process_remote_item(
                sp, item, idx, len(regular_items), is_register=False, force=force, result=result
            )

        logger.info(
            "SharePoint ETL complete: processed=%d skipped=%d failed=%d limitations=%d",
            result.processed_files,
            result.skipped_unchanged_files,
            result.failed_files,
            result.limitations_logged,
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
    ) -> None:
        result.discovered_files += 1
        rel_path = Path(item.relative_path)
        rel_posix = rel_path.as_posix()
        pass_label = "Pass 1" if is_register else "Pass 2"
        local_path: Path | None = None
        seeded_json: Path | None = None

        try:
            up_to_date = not force and self.manifest.is_remote_up_to_date(
                rel_posix,
                remote_etag=item.etag or item.ctag,
                quick_xor_hash=item.quick_xor_hash,
                sha256=item.sha256_hash,
            )
            if up_to_date:
                if is_register:
                    local_path = sp.download_file(item)
                    self.register_index.load_register_file(local_path)
                result.skipped_unchanged_files += 1
                logger.debug("[%s: %d/%d] SKIP unchanged: %s", pass_label, idx, total, item.name)
                return

            logger.debug(
                "[%s: %d/%d] Streaming & extracting: %s", pass_label, idx, total, rel_posix
            )
            local_path, seeded_json, file_hash = sp.download_and_seed(
                item, json_out_dir=self.json_out_dir
            )
            if is_register:
                self.register_index.load_register_file(local_path)

            seeded_doc = ParsedDoc.model_validate_json(seeded_json.read_text(encoding="utf-8"))
            parsed_doc = self._dispatch(
                local_path,
                rel_path,
                sharepoint_url=seeded_doc.metadata.sharepoint_url or item.web_url,
            )
            parsed_doc.document_id = seeded_doc.document_id
            parsed_doc.content_hash = file_hash
            parsed_doc.metadata.extra["sharepoint_item_id"] = seeded_doc.metadata.extra.get(
                "sharepoint_item_id"
            )
            parsed_doc.metadata.extra["sharepoint_etag"] = seeded_doc.metadata.extra.get(
                "sharepoint_etag"
            )
            if seeded_doc.metadata.sharepoint_url:
                parsed_doc.metadata.sharepoint_url = seeded_doc.metadata.sharepoint_url

            if self.quality_logger.evaluate_and_log(parsed_doc) is not None:
                result.limitations_logged += 1

            md_path, json_path = self._write_outputs(parsed_doc, json_path=seeded_json)
            self.manifest.record_success(
                rel_posix, parsed_doc, md_path, json_path, quick_xor_hash=item.quick_xor_hash
            )
            self.manifest.save()
            result.processed_files += 1
            result.documents.append(parsed_doc)
            logger.debug(
                "[%s: %d/%d] DONE: %s -> %s", pass_label, idx, total, item.name, json_path.name
            )

        except Exception as exc:
            if seeded_json is not None and seeded_json.exists():
                seeded_json.unlink(missing_ok=True)
            result.failed_files += 1
            logger.error("[%s: %d/%d] ERROR on %s: %s", pass_label, idx, total, item.name, exc)
            matched = self.metadata_matcher.match(rel_path, sharepoint_url=item.web_url)
            self.quality_logger.log_corrupted_file(
                rel_path,
                str(exc),
                project=matched.project,
                building=matched.building,
                phase=matched.phase,
            )
            result.limitations_logged += 1
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
            return self.pdf_extractor.extract(file_path, **kwargs)
        if ext == ".docx":
            return self.docx_extractor.extract(file_path, **kwargs)
        if ext in {".xlsx", ".xls"}:
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
