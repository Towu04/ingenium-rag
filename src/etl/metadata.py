"""3-Layer Metadata Matcher resolving canonical `domain.DocumentMetadata` directly.

Resolution priority (highest accuracy to fallback):
  1. Layer 1: Excel/CSV Document Register (`Documentenlijst` / `Borderel`)
  2. Layer 2: Master Project Catalog (`projects_catalog.csv`)
  3. Layer 2.5: SharePoint site URL (`/sites/12345001-INT/...`)
  4. Layer 3: Folder path & filename Flemish engineering heuristics
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import get_settings
from domain import (
    DocCategory,
    DocumentMetadata,
    ExtractionMethod,
    compute_doc_id,
    infer_building,
    infer_doc_category,
    infer_phase,
    infer_project_id_from_path,
    infer_project_id_from_sharepoint_url,
    normalize_building,
    normalize_phase,
    normalize_project,
    split_buildings,
    utc_now,
)
from etl.catalog import DocumentRegisterIndex, ProjectCatalog


@dataclass
class MatchedMetadata:
    """Intermediate resolution result used for inventory previews and building `DocumentMetadata`."""

    project: str | None = None
    project_name: str | None = field(default_factory=lambda: get_settings().default_project_name)
    sharepoint_url: str | None = None
    building: str | None = None
    phase: str | None = None
    doc_category: DocCategory = DocCategory.UNKNOWN
    revision: str | None = None
    doc_code: str | None = None
    resolution_sources: list[str] = field(default_factory=list)


class MetadataMatcher:
    """Layered domain metadata resolver emitting canonical `domain.DocumentMetadata`."""

    def __init__(
        self,
        catalog: ProjectCatalog | None = None,
        register_index: DocumentRegisterIndex | None = None,
    ) -> None:
        self.catalog = catalog or ProjectCatalog()
        self.register_index = register_index or DocumentRegisterIndex()

    def match(
        self,
        relative_file_path: str | Path,
        *,
        sharepoint_url: str | None = None,
    ) -> MatchedMetadata:
        rel_path = Path(relative_file_path)
        rel_posix = rel_path.as_posix()
        parent_posix = rel_path.parent.as_posix() if rel_path.parent != Path(".") else ""

        settings = get_settings()
        default_name = settings.default_project_name
        aliases = settings.building_aliases or None
        project: str | None = None
        project_name: str | None = default_name
        resolved_sp_url = sharepoint_url
        building: str | None = None
        phase: str | None = None
        doc_category = DocCategory.UNKNOWN
        revision: str | None = None
        doc_code: str | None = None
        sources: list[str] = []

        # Layer 3: Path & filename regex heuristics
        if inferred_proj := infer_project_id_from_path(rel_path):
            project = inferred_proj
            sources.append("layer3_regex_project")

        # Layer 2.5: SharePoint site slug overrides noisy filename heuristics
        if sp_proj := infer_project_id_from_sharepoint_url(sharepoint_url):
            project = sp_proj
            sources.append("layer2_5_sharepoint_site_project")

        folder_phase = infer_phase(parent_posix)
        file_phase = infer_phase(rel_path.name)
        if folder_phase:
            phase = folder_phase
            sources.append("layer3_regex_folder_phase")
        elif file_phase:
            phase = file_phase
            sources.append("layer3_regex_file_phase")

        if inferred_bldg := infer_building(rel_posix, extra_aliases=aliases):
            building = inferred_bldg
            sources.append("layer3_regex_building")

        inferred_cat = infer_doc_category(rel_path)
        if inferred_cat != DocCategory.UNKNOWN:
            doc_category = inferred_cat
            sources.append("layer3_regex_category")

        # Layer 2: Master Project Catalog
        catalog_entry = self.catalog.match_path(rel_path)
        if catalog_entry is not None:
            project = normalize_project(catalog_entry.project_id)
            project_name = catalog_entry.project_name or default_name
            resolved_sp_url = resolved_sp_url or catalog_entry.sharepoint_url
            if building is None and catalog_entry.building:
                building = normalize_building(catalog_entry.building) or catalog_entry.building
            if phase is None and catalog_entry.phase:
                phase = normalize_phase(infer_phase(catalog_entry.phase) or catalog_entry.phase)
            sources.append("layer2_catalog")

        # Layer 1: Excel/CSV Document Register (highest precedence)
        reg_entry = self.register_index.match_file(rel_path)
        if reg_entry is not None:
            if reg_entry.building:
                building = normalize_building(reg_entry.building) or reg_entry.building
            if reg_entry.phase:
                phase = normalize_phase(infer_phase(reg_entry.phase) or reg_entry.phase)
            if reg_entry.doc_category and reg_entry.doc_category != DocCategory.UNKNOWN:
                doc_category = reg_entry.doc_category
            if reg_entry.revision:
                revision = reg_entry.revision
            if reg_entry.doc_code:
                doc_code = reg_entry.doc_code
            sources.append("layer1_document_register")

        return MatchedMetadata(
            project=project,
            project_name=project_name,
            sharepoint_url=resolved_sp_url,
            building=building,
            phase=phase,
            doc_category=doc_category,
            revision=revision,
            doc_code=doc_code,
            resolution_sources=sources,
        )

    def build_document_metadata(
        self,
        file_path: str | Path,
        relative_file_path: str | Path,
        extraction_method: ExtractionMethod,
        *,
        file_hash_sha256: str | None = None,
        is_scanned: bool = False,
        is_hvac_diagram: bool = False,
        needs_ocr: bool = False,
        extraction_confidence: float = 1.0,
        quality_warnings: list[str] | None = None,
        dropped_noisy_callouts: int = 0,
        sharepoint_item_id: str | None = None,
        sharepoint_etag: str | None = None,
        sharepoint_url: str | None = None,
    ) -> tuple[str, str, DocumentMetadata]:
        """Return `(document_id, content_hash, DocumentMetadata)` ready for `ParsedDoc`."""
        actual_path = Path(file_path)
        rel_path = Path(relative_file_path)
        source_path = rel_path.as_posix()

        if not file_hash_sha256:
            if actual_path.exists():
                file_hash_sha256 = hashlib.sha256(actual_path.read_bytes()).hexdigest()
            else:
                file_hash_sha256 = hashlib.sha256(source_path.encode("utf-8")).hexdigest()

        doc_id = compute_doc_id(rel_path)
        matched = self.match(rel_path, sharepoint_url=sharepoint_url)
        buildings = split_buildings(matched.building)

        extra: dict[str, Any] = {
            "file_name": rel_path.name,
            "file_extension": rel_path.suffix.lower(),
            "relative_folder_path": rel_path.parent.as_posix()
            if rel_path.parent != Path(".")
            else "",
            "project_name": matched.project_name,
            "doc_category": matched.doc_category.value,
            "extraction_method": extraction_method.value,
            "is_scanned": is_scanned,
            "is_hvac_diagram": is_hvac_diagram,
            "needs_ocr": needs_ocr,
            "extraction_confidence": max(0.0, min(1.0, float(extraction_confidence))),
            "quality_warnings": list(quality_warnings or []),
            "dropped_noisy_callouts": dropped_noisy_callouts,
            "resolution_sources": matched.resolution_sources,
            "sharepoint_item_id": sharepoint_item_id,
            "sharepoint_etag": sharepoint_etag,
            "processed_at": utc_now().isoformat(),
        }
        if matched.revision:
            extra["revision"] = matched.revision
        if matched.doc_code:
            extra["doc_code"] = matched.doc_code

        metadata = DocumentMetadata(
            source_path=source_path,
            sharepoint_url=sharepoint_url or matched.sharepoint_url,
            project=matched.project,
            building=" - ".join(buildings) if buildings else None,
            buildings=buildings,
            phase=matched.phase,
            extra=extra,
        )
        return doc_id, file_hash_sha256, metadata
