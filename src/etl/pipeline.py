"""Incremental ETL pipeline orchestrator, SharePoint streaming coordinator, and SHA-256 manifest manager."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from etl.catalog import DocumentRegisterIndex, ProjectCatalog, is_document_register_file
from etl.config import (
    JSON_OUT_DIR,
    LIMITATIONS_REPORT_PATH,
    MANIFEST_PATH,
    MARKDOWN_OUT_DIR,
    SUPPORTED_EXTRACTION_EXTENSIONS,
)
from etl.extractors import DocxExtractor, PdfExtractor, XlsxExtractor
from etl.metadata_matcher import MetadataMatcher
from etl.quality import QualityLogger
from etl.schemas import (
    DocumentMetadata,
    ExtractedDocument,
    MatchedMetadata,
    coerce_unknown_to_none,
)
from etl.sharepoint_access import RemoteDriveItem, SharePointConnector, compute_file_sha256


class ManifestEntry(BaseModel):
    """State record for a single processed file inside `manifest.json`."""

    relative_path: str
    doc_id: str
    sha256: str
    file_name: str
    project_id: Optional[str] = None
    building: Optional[str] = None
    phase: Optional[str] = None
    doc_category: str
    markdown_path: str
    json_path: str
    page_count: int = 0
    table_count: int = 0
    block_count: int = 0
    extraction_confidence: float = 1.0
    is_scanned: bool = False
    is_hvac_diagram: bool = False
    needs_ocr: bool = False
    sharepoint_item_id: Optional[str] = None
    sharepoint_etag: Optional[str] = None
    quick_xor_hash: Optional[str] = None
    processed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("project_id", "building", "phase", mode="before")
    @classmethod
    def _coerce_sentinels(cls, value: Any) -> Any:
        return coerce_unknown_to_none(value)


class ManifestState(BaseModel):
    """Top-level structure persisted to `data/processed/manifest.json`."""

    version: str = "1.0"
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    files: Dict[str, ManifestEntry] = Field(default_factory=dict)


class ManifestManager:
    """Manages incremental SHA-256 and SharePoint eTag/quickXorHash state tracking in `manifest.json`."""

    def __init__(self, manifest_path: Union[str, Path] = MANIFEST_PATH) -> None:
        self.manifest_path = Path(manifest_path)
        self.state: ManifestState = self._load()

    def _load(self) -> ManifestState:
        if not self.manifest_path.exists():
            return ManifestState()
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return ManifestState.model_validate(raw)
        except Exception:
            return ManifestState()

    def save(self) -> None:
        """Persist current manifest state to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        self.manifest_path.write_text(
            self.state.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def _outputs_exist(self, entry: ManifestEntry) -> bool:
        if Path(entry.markdown_path).exists() and Path(entry.json_path).exists():
            return True
        # Portable fallback: resolve `<doc_id>.md` / `<doc_id>.json` relative to manifest's folder
        # so manifests generated on Windows still work on macOS/Linux.
        base_dir = self.manifest_path.parent
        return (base_dir / "markdown" / f"{entry.doc_id}.md").exists() and (
            base_dir / "json" / f"{entry.doc_id}.json"
        ).exists()

    def is_up_to_date(self, relative_path: str, file_sha256: str) -> bool:
        """Return True if the file's SHA-256 matches the manifest and output files exist."""
        norm_key = Path(relative_path).as_posix()
        entry = self.state.files.get(norm_key)
        if entry is None:
            return False
        if entry.sha256 != file_sha256:
            return False
        return self._outputs_exist(entry)

    def is_remote_up_to_date(
        self,
        relative_path: str,
        remote_etag: Optional[str] = None,
        quick_xor_hash: Optional[str] = None,
        sha256: Optional[str] = None,
    ) -> bool:
        """Pre-download check: return True if remote eTag/quickXorHash/sha256 matches manifest and outputs exist."""
        norm_key = Path(relative_path).as_posix()
        entry = self.state.files.get(norm_key)
        if entry is None or not self._outputs_exist(entry):
            return False

        if remote_etag and entry.sharepoint_etag and entry.sharepoint_etag == remote_etag:
            return True
        if quick_xor_hash and entry.quick_xor_hash and entry.quick_xor_hash == quick_xor_hash:
            return True
        if sha256 and entry.sha256 == sha256:
            return True
        return False

    def record_success(
        self,
        relative_path: str,
        doc: ExtractedDocument,
        markdown_path: Path,
        json_path: Path,
        quick_xor_hash: Optional[str] = None,
    ) -> ManifestEntry:
        """Record a newly extracted document in the manifest."""
        norm_key = Path(relative_path).as_posix()
        pages = {
            b.page_number for b in doc.blocks if b.page_number is not None
        }
        table_count = sum(1 for b in doc.blocks if b.block_type == "table")

        entry = ManifestEntry(
            relative_path=norm_key,
            doc_id=doc.metadata.doc_id,
            sha256=doc.metadata.file_hash_sha256,
            file_name=doc.metadata.file_name,
            project_id=doc.metadata.project_id,
            building=doc.metadata.building,
            phase=doc.metadata.phase,
            doc_category=doc.metadata.doc_category.value,
            markdown_path=str(markdown_path),
            json_path=str(json_path),
            page_count=len(pages),
            table_count=table_count,
            block_count=len(doc.blocks),
            extraction_confidence=doc.metadata.extraction_confidence,
            is_scanned=doc.metadata.is_scanned,
            is_hvac_diagram=doc.metadata.is_hvac_diagram,
            needs_ocr=doc.metadata.needs_ocr,
            sharepoint_item_id=doc.metadata.sharepoint_item_id,
            sharepoint_etag=doc.metadata.sharepoint_etag,
            quick_xor_hash=quick_xor_hash,
        )
        self.state.files[norm_key] = entry
        return entry


@dataclass
class InventoryItem:
    """Preview item returned by `ETLPipeline.scan_inventory`."""

    relative_path: str
    file_name: str
    extension: str
    size_bytes: int
    supported_for_extraction: bool
    matched_metadata: MatchedMetadata


@dataclass
class InventorySummary:
    """Summary returned by `ETLPipeline.scan_inventory`."""

    total_files: int = 0
    extension_counts: Dict[str, int] = field(default_factory=dict)
    items: List[InventoryItem] = field(default_factory=list)


@dataclass
class PipelineRunResult:
    """Execution statistics from an incremental `ETLPipeline.run` invocation."""

    discovered_files: int = 0
    processed_files: int = 0
    skipped_unchanged_files: int = 0
    failed_files: int = 0
    limitations_logged: int = 0
    extracted_documents: List[ExtractedDocument] = field(default_factory=list)


class ETLPipeline:
    """Incremental ETL, Metadata Matching, and Document Extraction pipeline."""

    def __init__(
        self,
        catalog_path: Optional[Union[str, Path]] = None,
        markdown_out_dir: Union[str, Path] = MARKDOWN_OUT_DIR,
        json_out_dir: Union[str, Path] = JSON_OUT_DIR,
        manifest_path: Union[str, Path] = MANIFEST_PATH,
        limitations_report_path: Union[str, Path] = LIMITATIONS_REPORT_PATH,
    ) -> None:
        self.markdown_out_dir = Path(markdown_out_dir)
        self.json_out_dir = Path(json_out_dir)
        self.manifest = ManifestManager(manifest_path=manifest_path)
        self.quality_logger = QualityLogger(report_path=limitations_report_path)

        catalog = (
            ProjectCatalog.from_file(catalog_path)
            if catalog_path and Path(catalog_path).exists()
            else ProjectCatalog()
        )
        self.register_index = DocumentRegisterIndex()
        self.metadata_matcher = MetadataMatcher(
            catalog=catalog,
            register_index=self.register_index,
        )

        self.pdf_extractor = PdfExtractor(metadata_matcher=self.metadata_matcher)
        self.docx_extractor = DocxExtractor(metadata_matcher=self.metadata_matcher)
        self.xlsx_extractor = XlsxExtractor(metadata_matcher=self.metadata_matcher)

    def scan_inventory(self, raw_dir: Union[str, Path]) -> InventorySummary:
        """Scan a local SharePoint directory tree, count extensions, and preview inferred metadata without OCR."""
        root = Path(raw_dir)
        if not root.exists():
            raise FileNotFoundError(f"Raw SharePoint directory not found: {root}")

        # Pre-load any document registers (Layer 1) before previewing sibling files
        self.register_index.discover_and_load(root)

        summary = InventorySummary()
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue

            rel_path = file_path.relative_to(root)
            ext = file_path.suffix.lower() or "(no_ext)"
            summary.total_files += 1
            summary.extension_counts[ext] = summary.extension_counts.get(ext, 0) + 1

            matched = self.metadata_matcher.match(rel_path)
            summary.items.append(
                InventoryItem(
                    relative_path=rel_path.as_posix(),
                    file_name=file_path.name,
                    extension=ext,
                    size_bytes=file_path.stat().st_size,
                    supported_for_extraction=ext in SUPPORTED_EXTRACTION_EXTENSIONS,
                    matched_metadata=matched,
                )
            )
        return summary

    def scan_sharepoint_inventory(
        self,
        connector: Optional[SharePointConnector] = None,
    ) -> InventorySummary:
        """List remote SharePoint files and preview 3-layer metadata without downloading heavy binaries."""
        sp = connector or SharePointConnector.from_env(json_out_dir=self.json_out_dir)
        remote_items = sp.list_remote_files(supported_only=False)

        # Temporarily download and load only lightweight Document Registers (Layer 1), then delete immediately
        for item in remote_items:
            if is_document_register_file(item.relative_path):
                local_reg: Optional[Path] = None
                try:
                    local_reg = sp.download_file(item)
                    self.register_index.load_register_file(local_reg)
                except Exception:
                    pass
                finally:
                    if local_reg is not None:
                        sp.cleanup_local_file(local_reg)

        summary = InventorySummary()
        for item in remote_items:
            ext = item.extension or "(no_ext)"
            summary.total_files += 1
            summary.extension_counts[ext] = summary.extension_counts.get(ext, 0) + 1

            matched = self.metadata_matcher.match(
                item.relative_path,
                sharepoint_url=item.web_url,
            )
            if item.web_url and not matched.sharepoint_url:
                matched.sharepoint_url = item.web_url

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

    def run(
        self,
        raw_dir: Union[str, Path],
        force: bool = False,
    ) -> PipelineRunResult:
        """Execute incremental document extraction across a local SharePoint project folder tree."""
        root = Path(raw_dir)
        if not root.exists():
            raise FileNotFoundError(f"Raw SharePoint directory not found: {root}")

        self.markdown_out_dir.mkdir(parents=True, exist_ok=True)
        self.json_out_dir.mkdir(parents=True, exist_ok=True)

        # Pass 1: Discover and index all Excel/CSV Document Registers (Layer 1) first
        print(f"[Pass 1] Scanning local directory for Document Registers: {root}", flush=True)
        loaded_regs = self.register_index.discover_and_load(root)
        print(f"[Pass 1] Indexed {loaded_regs} Document Register entries.", flush=True)

        result = PipelineRunResult()
        supported_files = [
            p
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in SUPPORTED_EXTRACTION_EXTENSIONS
        ]
        total_supported = len(supported_files)
        print(f"[Pass 2] Processing {total_supported} supported local files...", flush=True)

        # Pass 2: Incrementally process all supported files
        for idx, file_path in enumerate(supported_files, start=1):
            result.discovered_files += 1
            rel_path = file_path.relative_to(root)
            rel_posix = rel_path.as_posix()
            file_hash = compute_file_sha256(file_path)

            if not force and self.manifest.is_up_to_date(rel_posix, file_hash):
                result.skipped_unchanged_files += 1
                print(f"[{idx}/{total_supported}] SKIP (unchanged): {rel_posix}", flush=True)
                continue

            try:
                print(f"[{idx}/{total_supported}] Extracting: {rel_posix} ...", flush=True)
                extracted_doc = self._dispatch_extractor(file_path, rel_path)
                extracted_doc.metadata.file_hash_sha256 = file_hash

                limitation = self.quality_logger.evaluate_and_log(extracted_doc)
                if limitation is not None:
                    result.limitations_logged += 1

                md_path, json_path = self._write_outputs(extracted_doc)
                self.manifest.record_success(
                    relative_path=rel_posix,
                    doc=extracted_doc,
                    markdown_path=md_path,
                    json_path=json_path,
                )
                self.manifest.save()

                result.processed_files += 1
                result.extracted_documents.append(extracted_doc)
                print(
                    f"[{idx}/{total_supported}] DONE: {file_path.name} -> {md_path.name} ({len(extracted_doc.blocks)} blocks)",
                    flush=True,
                )

            except Exception as exc:
                result.failed_files += 1
                print(f"[{idx}/{total_supported}] ERROR: {rel_posix} ({exc})", flush=True)
                matched = self.metadata_matcher.match(rel_path)
                self.quality_logger.log_corrupted_file(
                    relative_path=rel_path,
                    error_message=str(exc),
                    project_id=matched.project_id,
                    building=matched.building,
                    phase=matched.phase,
                )
                result.limitations_logged += 1

        return result

    def run_from_sharepoint(
        self,
        connector: Optional[SharePointConnector] = None,
        force: bool = False,
    ) -> PipelineRunResult:
        """Stream files from SharePoint one by one, enrich seeded JSON in-place, and delete raw files immediately."""
        sp = connector or SharePointConnector.from_env(json_out_dir=self.json_out_dir)

        self.markdown_out_dir.mkdir(parents=True, exist_ok=True)
        self.json_out_dir.mkdir(parents=True, exist_ok=True)

        remote_items = sp.list_remote_files(supported_only=True)
        register_items = [
            item for item in remote_items if is_document_register_file(item.relative_path)
        ]
        regular_items = [
            item for item in remote_items if not is_document_register_file(item.relative_path)
        ]
        total_remote = len(remote_items)
        print(
            f"[Pipeline] Starting 2-pass extraction: {len(register_items)} Document Register(s) (Pass 1) + {len(regular_items)} regular file(s) (Pass 2) = {total_remote} total.",
            flush=True,
        )

        result = PipelineRunResult()

        # Pass 1 — Document Registers First:
        # Download temporarily -> parse into DocumentRegisterIndex -> extract .md & in-place .json -> delete local .xlsx
        for idx, item in enumerate(register_items, start=1):
            result.discovered_files += 1
            rel_path = Path(item.relative_path)
            rel_posix = rel_path.as_posix()
            local_path: Optional[Path] = None
            seeded_json_path: Optional[Path] = None

            try:
                already_up_to_date = not force and self.manifest.is_remote_up_to_date(
                    rel_posix,
                    remote_etag=item.etag or item.ctag,
                    quick_xor_hash=item.quick_xor_hash,
                    sha256=item.sha256_hash,
                )
                if already_up_to_date:
                    print(
                        f"[Pass 1: {idx}/{len(register_items)}] Loading cached Register (unchanged): {item.name}",
                        flush=True,
                    )
                    # Still load into register_index for Layer 1 matching of new sibling files in Pass 2
                    local_path = sp.download_file(item)
                    self.register_index.load_register_file(local_path)
                    result.skipped_unchanged_files += 1
                    continue

                print(
                    f"[Pass 1: {idx}/{len(register_items)}] Downloading & indexing Register: {rel_posix} ...",
                    flush=True,
                )
                local_path, seeded_json_path, file_hash = sp.download_and_seed(
                    item,
                    json_out_dir=self.json_out_dir,
                )
                self.register_index.load_register_file(local_path)

                self._enrich_seeded_file(
                    local_path=local_path,
                    rel_path=rel_path,
                    seeded_json_path=seeded_json_path,
                    file_hash=file_hash,
                    remote_item=item,
                    result=result,
                )
                print(
                    f"[Pass 1: {idx}/{len(register_items)}] DONE: {item.name} -> {seeded_json_path.name}",
                    flush=True,
                )
            except Exception as exc:
                if seeded_json_path is not None and seeded_json_path.exists():
                    seeded_json_path.unlink(missing_ok=True)
                result.failed_files += 1
                print(
                    f"[Pass 1: {idx}/{len(register_items)}] ERROR: {item.name} ({exc})",
                    flush=True,
                )
                matched = self.metadata_matcher.match(rel_path)
                self.quality_logger.log_corrupted_file(
                    relative_path=rel_path,
                    error_message=str(exc),
                    project_id=matched.project_id,
                    building=matched.building,
                    phase=matched.phase,
                )
                result.limitations_logged += 1
            finally:
                if local_path is not None:
                    sp.cleanup_local_file(local_path)

        # Pass 2 — Stream, Extract In-Place, and Delete Per File:
        for idx, item in enumerate(regular_items, start=1):
            result.discovered_files += 1
            rel_path = Path(item.relative_path)
            rel_posix = rel_path.as_posix()

            # Pre-Download Manifest Check
            if not force and self.manifest.is_remote_up_to_date(
                rel_posix,
                remote_etag=item.etag or item.ctag,
                quick_xor_hash=item.quick_xor_hash,
                sha256=item.sha256_hash,
            ):
                result.skipped_unchanged_files += 1
                print(
                    f"[Pass 2: {idx}/{len(regular_items)}] SKIP (unchanged): {item.name}",
                    flush=True,
                )
                continue

            local_path = None
            seeded_json_path = None
            try:
                print(
                    f"[Pass 2: {idx}/{len(regular_items)}] Streaming & extracting: {rel_posix} ...",
                    flush=True,
                )
                # Step A: Download & Seed Target JSON
                local_path, seeded_json_path, file_hash = sp.download_and_seed(
                    item,
                    json_out_dir=self.json_out_dir,
                )

                # Step B: Enrich the Existing JSON in-place & Write Markdown
                self._enrich_seeded_file(
                    local_path=local_path,
                    rel_path=rel_path,
                    seeded_json_path=seeded_json_path,
                    file_hash=file_hash,
                    remote_item=item,
                    result=result,
                )
                print(
                    f"[Pass 2: {idx}/{len(regular_items)}] DONE: {item.name} -> {seeded_json_path.name} (raw file deleted)",
                    flush=True,
                )
            except Exception as exc:
                if seeded_json_path is not None and seeded_json_path.exists():
                    seeded_json_path.unlink(missing_ok=True)
                result.failed_files += 1
                print(
                    f"[Pass 2: {idx}/{len(regular_items)}] ERROR: {item.name} ({exc})",
                    flush=True,
                )
                matched = self.metadata_matcher.match(rel_path)
                self.quality_logger.log_corrupted_file(
                    relative_path=rel_path,
                    error_message=str(exc),
                    project_id=matched.project_id,
                    building=matched.building,
                    phase=matched.phase,
                )
                result.limitations_logged += 1
            finally:
                # Step C: Delete Local Raw Binary immediately
                if local_path is not None:
                    sp.cleanup_local_file(local_path)

        print(
            f"[Pipeline] Complete! Processed: {result.processed_files} | Skipped: {result.skipped_unchanged_files} | Failed: {result.failed_files}",
            flush=True,
        )
        return result

    def _enrich_seeded_file(
        self,
        local_path: Path,
        rel_path: Path,
        seeded_json_path: Path,
        file_hash: str,
        remote_item: RemoteDriveItem,
        result: PipelineRunResult,
    ) -> None:
        """Load pre-seeded JSON, merge 3-layer metadata + extracted blocks, and overwrite that exact JSON."""
        raw_seeded = json.loads(seeded_json_path.read_text(encoding="utf-8"))
        if "metadata" in raw_seeded:
            seeded_meta = DocumentMetadata.model_validate(raw_seeded["metadata"])
        else:
            seeded_meta = DocumentMetadata.model_validate(raw_seeded)

        extracted_doc = self._dispatch_extractor(
            local_path,
            rel_path,
            sharepoint_url=seeded_meta.sharepoint_url or remote_item.web_url,
        )

        # Preserve SharePoint provenance seeded in Step A while merging 3-layer resolved metadata
        extracted_doc.metadata.doc_id = seeded_meta.doc_id
        extracted_doc.metadata.file_hash_sha256 = file_hash
        extracted_doc.metadata.sharepoint_item_id = seeded_meta.sharepoint_item_id
        extracted_doc.metadata.sharepoint_etag = seeded_meta.sharepoint_etag
        if seeded_meta.sharepoint_url:
            extracted_doc.metadata.sharepoint_url = seeded_meta.sharepoint_url

        limitation = self.quality_logger.evaluate_and_log(extracted_doc)
        if limitation is not None:
            result.limitations_logged += 1

        md_path, json_path = self._write_outputs(
            extracted_doc,
            json_path=seeded_json_path,
        )
        self.manifest.record_success(
            relative_path=rel_path.as_posix(),
            doc=extracted_doc,
            markdown_path=md_path,
            json_path=json_path,
            quick_xor_hash=remote_item.quick_xor_hash,
        )
        self.manifest.save()

        result.processed_files += 1
        result.extracted_documents.append(extracted_doc)

    def _dispatch_extractor(
        self,
        file_path: Path,
        rel_path: Path,
        sharepoint_url: Optional[str] = None,
    ) -> ExtractedDocument:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self.pdf_extractor.extract(
                file_path, relative_path=rel_path, sharepoint_url=sharepoint_url
            )
        if ext == ".docx":
            return self.docx_extractor.extract(
                file_path, relative_path=rel_path, sharepoint_url=sharepoint_url
            )
        if ext in {".xlsx", ".xls"}:
            return self.xlsx_extractor.extract(
                file_path, relative_path=rel_path, sharepoint_url=sharepoint_url
            )
        raise ValueError(f"Unsupported file extension for extraction: {ext}")

    def _write_outputs(
        self,
        doc: ExtractedDocument,
        json_path: Optional[Path] = None,
        md_path: Optional[Path] = None,
    ) -> tuple[Path, Path]:
        doc_id = doc.metadata.doc_id
        target_md = md_path or (self.markdown_out_dir / f"{doc_id}.md")
        target_json = json_path or (self.json_out_dir / f"{doc_id}.json")

        target_md.parent.mkdir(parents=True, exist_ok=True)
        target_json.parent.mkdir(parents=True, exist_ok=True)

        target_md.write_text(doc.full_markdown_with_anchors, encoding="utf-8")
        target_json.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        return target_md, target_json

    def get_summary_report(self) -> Dict[str, Any]:
        """Compile extraction metrics and logged limitations for the weekly Ingenium review."""
        entries = list(self.manifest.state.files.values())
        limitations = self.quality_logger.read_records()

        total_pages = sum(e.page_count for e in entries)
        total_tables = sum(e.table_count for e in entries)
        total_blocks = sum(e.block_count for e in entries)
        by_category: Dict[str, int] = {}
        for e in entries:
            by_category[e.doc_category] = by_category.get(e.doc_category, 0) + 1

        return {
            "total_documents": len(entries),
            "total_pages": total_pages,
            "total_tables": total_tables,
            "total_blocks": total_blocks,
            "by_category": by_category,
            "entries": entries,
            "limitations": limitations,
        }
