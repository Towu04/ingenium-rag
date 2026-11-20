"""Adapter translating `src/etl` outputs (`ExtractedDocument`) into core `domain.ParsedDoc` contracts.

This is the sole boundary module that imports both `etl.schemas` and `domain`, keeping
the ETL package cleanly decoupled from downstream chunking, embedding, and retrieval.
"""

import json
import logging
from pathlib import Path
from typing import Any

from domain import DocumentMetadata as DomainMetadata
from domain import ParsedDoc, ParsedSection
from etl.metadata_matcher import (
    infer_phase as etl_infer_phase,
    infer_project_id_from_sharepoint_url,
)
from etl.pipeline import ManifestState
from etl.quality import score_callout_readability
from etl.schemas import ContentBlock, ExtractedDocument as EtlExtractedDocument
from processing.normalization import (
    is_bogus_heading,
    normalize_phase,
    normalize_project,
    split_buildings,
)

logger = logging.getLogger(__name__)

DEFAULT_CALLOUT_READABILITY_THRESHOLD = 0.45


def _build_source_path(relative_folder_path: str, file_name: str) -> str:
    folder = relative_folder_path.strip().strip("/")
    return f"{folder}/{file_name}" if folder else file_name


def _resolve_project(etl_doc: EtlExtractedDocument) -> str | None:
    meta = etl_doc.metadata
    sp_project = infer_project_id_from_sharepoint_url(meta.sharepoint_url)
    raw_project = normalize_project(meta.project_id)
    # SharePoint site slug (`/sites/06038004-INT`) is authoritative over legacy filename guesses
    # like supplier order numbers (`341-1883`) or document codes (`1003-004`).
    if sp_project:
        return normalize_project(sp_project)
    return raw_project


def _resolve_phase(etl_doc: EtlExtractedDocument, source_path: str) -> tuple[str | None, str | None]:
    """Return `(canonical_phase, demoted_raw_phase)`.

    Guards legacy JSON files where diagram body text (`DO` = Digital Output, `AB` = cable label)
    produced a false-positive phase that does not appear anywhere in the file path.
    """
    meta = etl_doc.metadata
    canonical = normalize_phase(meta.phase)
    if canonical is None:
        return None, None

    if meta.is_hvac_diagram and not meta.resolution_sources:
        path_phase = normalize_phase(etl_infer_phase(source_path))
        if path_phase is None:
            return None, meta.phase

    return canonical, None


def _clean_section_header(block: ContentBlock) -> str | None:
    candidate = block.section_heading or block.sheet_name
    if not candidate:
        return None
    if is_bogus_heading(candidate):
        return None
    return candidate.strip() or None


def adapt_extracted_document(
    etl_doc: EtlExtractedDocument,
    *,
    coalesce_prose: bool = True,
    drop_noisy_callouts: bool = True,
    callout_readability_threshold: float = DEFAULT_CALLOUT_READABILITY_THRESHOLD,
) -> ParsedDoc:
    """Convert an ETL `ExtractedDocument` into a clean, normalized `domain.ParsedDoc`."""
    meta = etl_doc.metadata
    source_path = _build_source_path(meta.relative_folder_path, meta.file_name)

    project = _resolve_project(etl_doc)
    buildings = split_buildings(meta.building)
    building = " - ".join(buildings) if buildings else None
    phase, demoted_raw_phase = _resolve_phase(etl_doc, source_path)

    sections: list[ParsedSection] = []
    dropped_noisy_callouts = 0

    for block in etl_doc.blocks:
        text = block.markdown_content.strip()
        if not text:
            continue

        if block.block_type == "diagram_callout" and drop_noisy_callouts:
            readability = score_callout_readability(text)
            if readability < callout_readability_threshold:
                dropped_noisy_callouts += 1
                continue

        header = _clean_section_header(block)
        pages = [block.page_number] if block.page_number is not None else []

        # Merge consecutive prose blocks sharing the same section header so sentences spanning
        # PDF page boundaries are not artificially broken into micro-sections.
        if (
            coalesce_prose
            and block.block_type == "text"
            and sections
            and sections[-1].section_header == header
            and not sections[-1].text.lstrip().startswith("|")
            and not text.lstrip().startswith("|")
        ):
            prev = sections[-1]
            merged_pages = sorted(set(prev.page_numbers) | set(pages))
            sections[-1] = ParsedSection(
                text=f"{prev.text}\n\n{text}",
                page_numbers=merged_pages,
                section_header=header,
            )
            continue

        sections.append(
            ParsedSection(
                text=text,
                page_numbers=pages,
                section_header=header,
            )
        )

    if dropped_noisy_callouts:
        logger.info(
            "Dropped %d noisy diagram_callout block(s) from '%s' (doc_id=%s).",
            dropped_noisy_callouts,
            meta.file_name,
            meta.doc_id,
        )

    extra: dict[str, Any] = {
        "file_name": meta.file_name,
        "file_extension": meta.file_extension,
        "project_name": meta.project_name,
        "doc_category": meta.doc_category.value,
        "extraction_method": meta.extraction_method.value,
        "is_scanned": meta.is_scanned,
        "is_hvac_diagram": meta.is_hvac_diagram,
        "needs_ocr": meta.needs_ocr or (meta.is_scanned and len(etl_doc.blocks) == 0),
        "extraction_confidence": meta.extraction_confidence,
        "quality_warnings": list(meta.quality_warnings),
        "sharepoint_item_id": meta.sharepoint_item_id,
        "sharepoint_etag": meta.sharepoint_etag,
        "processed_at": meta.processed_at.isoformat(),
        "dropped_noisy_callouts": dropped_noisy_callouts,
    }
    if meta.revision:
        extra["revision"] = meta.revision
    if meta.doc_code:
        extra["doc_code"] = meta.doc_code
    if demoted_raw_phase:
        extra["demoted_raw_phase"] = demoted_raw_phase

    domain_metadata = DomainMetadata(
        source_path=source_path,
        sharepoint_url=meta.sharepoint_url,
        project=project,
        building=building,
        buildings=buildings,
        phase=phase,
        extra=extra,
    )

    clean_markdown = "\n\n".join(s.text for s in sections)
    return ParsedDoc(
        document_id=meta.doc_id,
        content_hash=meta.file_hash_sha256,
        markdown=clean_markdown,
        sections=sections,
        metadata=domain_metadata,
    )


def load_parsed_doc_from_json(
    json_path: Path | str,
    *,
    coalesce_prose: bool = True,
    drop_noisy_callouts: bool = True,
    callout_readability_threshold: float = DEFAULT_CALLOUT_READABILITY_THRESHOLD,
) -> ParsedDoc | None:
    """Load an ETL `.json` file and adapt it to `ParsedDoc`.

    Returns `None` (with a warning log) if the file has 0 blocks (e.g. a scanned PDF needing OCR
    or an orphan seeded JSON left behind by an interrupted SharePoint run).
    """
    path = Path(json_path)
    etl_doc = EtlExtractedDocument.model_validate_json(path.read_text(encoding="utf-8"))
    if not etl_doc.blocks:
        if etl_doc.metadata.is_scanned or etl_doc.metadata.needs_ocr:
            logger.warning(
                "Skipping scanned PDF with 0 extracted blocks (needs_ocr=True): %s (%s)",
                etl_doc.metadata.file_name,
                path.name,
            )
        else:
            logger.warning(
                "Skipping empty/orphan seeded ETL JSON with 0 blocks: %s (%s)",
                etl_doc.metadata.file_name,
                path.name,
            )
        return None

    return adapt_extracted_document(
        etl_doc,
        coalesce_prose=coalesce_prose,
        drop_noisy_callouts=drop_noisy_callouts,
        callout_readability_threshold=callout_readability_threshold,
    )


def load_corpus_from_etl_dir(
    processed_dir: Path | str,
    *,
    use_manifest: bool = True,
    dedupe_by_hash: bool = True,
    coalesce_prose: bool = True,
    drop_noisy_callouts: bool = True,
    callout_readability_threshold: float = DEFAULT_CALLOUT_READABILITY_THRESHOLD,
) -> list[ParsedDoc]:
    """Load and adapt all valid ETL documents from a processed directory (`data/processed/`).

    Uses `manifest.json` when available (resolving JSON files portably by `doc_id` rather than
    OS-specific absolute paths) so orphan seeded JSON files are never ingested.
    """
    base = Path(processed_dir)
    json_dir = base / "json"
    manifest_path = base / "manifest.json"

    json_paths: list[Path] = []
    if use_manifest and manifest_path.exists():
        manifest = ManifestState.model_validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        for entry in manifest.files.values():
            portable_path = json_dir / f"{entry.doc_id}.json"
            if portable_path.exists():
                json_paths.append(portable_path)
            elif Path(entry.json_path).exists():
                json_paths.append(Path(entry.json_path))
            else:
                logger.warning(
                    "Manifest entry '%s' (doc_id=%s) missing JSON file at %s",
                    entry.file_name,
                    entry.doc_id,
                    portable_path,
                )
    elif json_dir.exists():
        json_paths = sorted(json_dir.glob("*.json"))
    else:
        raise FileNotFoundError(f"ETL processed JSON directory not found: {json_dir}")

    docs: list[ParsedDoc] = []
    seen_by_hash: dict[str, ParsedDoc] = {}
    total_dropped_callouts = 0

    for jp in json_paths:
        parsed = load_parsed_doc_from_json(
            jp,
            coalesce_prose=coalesce_prose,
            drop_noisy_callouts=drop_noisy_callouts,
            callout_readability_threshold=callout_readability_threshold,
        )
        if parsed is None:
            continue

        total_dropped_callouts += int(parsed.metadata.extra.get("dropped_noisy_callouts", 0))

        if dedupe_by_hash and parsed.content_hash in seen_by_hash:
            canonical_doc = seen_by_hash[parsed.content_hash]
            aliases = canonical_doc.metadata.extra.setdefault("duplicate_source_paths", [])
            aliases.append(parsed.metadata.source_path)
            logger.info(
                "Deduplicated byte-identical file '%s' (same SHA-256 as '%s').",
                parsed.metadata.source_path,
                canonical_doc.metadata.source_path,
            )
            continue

        seen_by_hash[parsed.content_hash] = parsed
        docs.append(parsed)

    logger.info(
        "Loaded %d ParsedDoc(s) from %s (%d noisy diagram_callout blocks quarantined).",
        len(docs),
        base,
        total_dropped_callouts,
    )
    return docs
