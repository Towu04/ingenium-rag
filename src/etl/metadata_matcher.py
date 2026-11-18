"""3-Layer Domain Metadata Matcher for AZ Sint-Jan Brugge engineering projects.

Resolution priority (highest accuracy to fallback):
  1. Layer 1: Excel Document Register Lookup (Documentenlijst / Index / Borderel)
  2. Layer 2: Master Google Sheet Project Catalog (projects_catalog.csv)
  3. Layer 3: SharePoint Folder Path & Filename Regex Heuristics (Flemish/Dutch engineering conventions)
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Union

from etl.catalog import DocumentRegisterIndex, ProjectCatalog
from etl.config import (
    DEFAULT_BUILDING,
    DEFAULT_PHASE,
    DEFAULT_PROJECT_ID,
    DEFAULT_PROJECT_NAME,
    HOSPITAL_ZONE_PATTERNS,
    PHASE_RULES,
    PRIMARY_DOC_CATEGORY_RULES,
    PROJECT_ID_PATTERNS,
    SECONDARY_DOC_CATEGORY_RULES,
    STRUCTURAL_BUILDING_PATTERNS,
)
from etl.schemas import (
    DocCategory,
    DocumentMetadata,
    ExtractionMethod,
    MatchedMetadata,
)


def compute_doc_id(relative_file_path: Union[str, Path]) -> str:
    """Compute a deterministic 16-character hex document ID from its relative POSIX path."""
    rel_posix = Path(relative_file_path).as_posix()
    return hashlib.sha256(rel_posix.encode("utf-8")).hexdigest()[:16]


def infer_phase(text: str) -> Optional[str]:
    """Infer canonical engineering project phase from folder path, filename, or register string."""
    if not text or not text.strip():
        return None
    for pattern, canonical_phase in PHASE_RULES:
        if pattern.search(text):
            return canonical_phase
    return None


def infer_building(text: str) -> Optional[str]:
    """Infer building, wing, floor, or hospital department zone from folder path or filename.

    Supports both structural tokens (e.g., 'Blok B', 'Vleugel Zuid', 'Gebouw 2', 'Verdieping +1',
    'Niv -1', 'Campus Sint-Jan') and hospital zones ('Operatiekwartier', 'Spoed', 'Labo',
    'Technische Ruimte'). When both a structural identifier and a functional zone appear in the
    path (e.g., 'Blok_B_Operatiekwartier'), combines them as 'Blok B - Operatiekwartier'.
    """
    if not text or not text.strip():
        return None

    matched_parts: List[str] = []

    for pattern, template in STRUCTURAL_BUILDING_PATTERNS:
        for match in pattern.finditer(text):
            raw_groups = [g.strip() for g in match.groups()]
            if template == "Niv {0}":
                formatted = f"Niv {raw_groups[0]}"
            elif template == "Verdieping {1}":
                formatted = f"Verdieping {raw_groups[1]}"
            else:
                prefix = raw_groups[0].replace("_", " ").replace("-", " ").strip().title()
                label = raw_groups[1].replace("_", " ").replace("-", " ").strip()
                # Uppercase single-letter/alphanumeric block codes (e.g., 'Blok b' -> 'Blok B')
                if len(label) <= 3 or any(ch.isdigit() for ch in label):
                    label = label.upper()
                else:
                    label = label.title()
                formatted = f"{prefix} {label}"
            if formatted not in matched_parts:
                matched_parts.append(formatted)

    for pattern, canonical_zone in HOSPITAL_ZONE_PATTERNS:
        if pattern.search(text):
            if canonical_zone not in matched_parts:
                matched_parts.append(canonical_zone)

    if not matched_parts:
        return None

    return " - ".join(matched_parts)


def infer_doc_category(file_path: Union[str, Path]) -> DocCategory:
    """Infer document category from filename (primary) and parent folder path (fallback)."""
    path = Path(file_path)
    filename = path.name
    full_path_str = str(path)

    # 1. Check primary document type keywords on filename first
    for pattern, category in PRIMARY_DOC_CATEGORY_RULES:
        if pattern.search(filename):
            return category

    # 2. Check secondary discipline/schedule keywords on filename
    for pattern, category in SECONDARY_DOC_CATEGORY_RULES:
        if pattern.search(filename):
            return category

    # 3. Fallback: inspect parent folder names
    for pattern, category in PRIMARY_DOC_CATEGORY_RULES:
        if pattern.search(full_path_str):
            return category

    for pattern, category in SECONDARY_DOC_CATEGORY_RULES:
        if pattern.search(full_path_str):
            return category

    return DocCategory.UNKNOWN


def infer_project_id_from_path(relative_path: Union[str, Path]) -> Optional[str]:
    """Extract a project identifier code from the top-level folder or path when catalog is absent."""
    path = Path(relative_path)
    parts = [p for p in path.parts if p not in {".", "/", "\\"}]
    for part in parts:
        for pattern in PROJECT_ID_PATTERNS:
            match = pattern.search(part)
            if match:
                return match.group(1).upper()
    return None


class MetadataMatcher:
    """Layered domain metadata resolver for AZ Sint-Jan Brugge engineering documents."""

    def __init__(
        self,
        catalog: Optional[ProjectCatalog] = None,
        register_index: Optional[DocumentRegisterIndex] = None,
    ) -> None:
        self.catalog = catalog or ProjectCatalog()
        self.register_index = register_index or DocumentRegisterIndex()

    def match(self, relative_file_path: Union[str, Path]) -> MatchedMetadata:
        """Resolve project, building, phase, and doc_category using Layers 1 -> 2 -> 3."""
        rel_path = Path(relative_file_path)
        rel_posix = rel_path.as_posix()
        parent_posix = rel_path.parent.as_posix() if rel_path.parent != Path(".") else ""

        project_id = DEFAULT_PROJECT_ID
        project_name = DEFAULT_PROJECT_NAME
        sharepoint_url: Optional[str] = None
        building = DEFAULT_BUILDING
        phase = DEFAULT_PHASE
        doc_category = DocCategory.UNKNOWN
        revision: Optional[str] = None
        doc_code: Optional[str] = None
        sources: List[str] = []

        # -------------------------------------------------------------------
        # Layer 3 (Base Fallback): Folder Path & Filename Regex Heuristics
        # -------------------------------------------------------------------
        inferred_proj_id = infer_project_id_from_path(rel_path)
        if inferred_proj_id:
            project_id = inferred_proj_id
            sources.append("layer3_regex_project_id")

        # Prefer phase inferred from folder structure over filename (so a 'Bestek' file
        # inside an 'Uitvoering' folder inherits 'Uitvoering', while a 'Bestek' file in
        # an unlabelled folder still resolves to 'Aanbesteding').
        folder_phase = infer_phase(parent_posix)
        file_phase = infer_phase(rel_path.name)
        if folder_phase:
            phase = folder_phase
            sources.append("layer3_regex_folder_phase")
        elif file_phase:
            phase = file_phase
            sources.append("layer3_regex_file_phase")

        inferred_bldg = infer_building(rel_posix)
        if inferred_bldg:
            building = inferred_bldg
            sources.append("layer3_regex_building")

        inferred_cat = infer_doc_category(rel_path)
        if inferred_cat != DocCategory.UNKNOWN:
            doc_category = inferred_cat
            sources.append("layer3_regex_category")

        # -------------------------------------------------------------------
        # Layer 2 (Catalog Override): Master Google Sheet Catalog
        # -------------------------------------------------------------------
        catalog_entry = self.catalog.match_path(rel_path)
        if catalog_entry is not None:
            project_id = catalog_entry.project_id
            project_name = catalog_entry.project_name or DEFAULT_PROJECT_NAME
            sharepoint_url = catalog_entry.sharepoint_url
            if building == DEFAULT_BUILDING and catalog_entry.building:
                building = infer_building(catalog_entry.building) or catalog_entry.building
            if phase == DEFAULT_PHASE and catalog_entry.phase:
                phase = infer_phase(catalog_entry.phase) or catalog_entry.phase
            sources.append("layer2_catalog")

        # -------------------------------------------------------------------
        # Layer 1 (Highest Accuracy Override): Excel Document Register
        # -------------------------------------------------------------------
        reg_entry = self.register_index.match_file(rel_path)
        if reg_entry is not None:
            if reg_entry.building:
                # Normalize if it matches standard building patterns, otherwise keep exact register value
                building = infer_building(reg_entry.building) or reg_entry.building
            if reg_entry.phase:
                phase = infer_phase(reg_entry.phase) or reg_entry.phase
            if reg_entry.doc_category and reg_entry.doc_category != DocCategory.UNKNOWN:
                doc_category = reg_entry.doc_category
            if reg_entry.revision:
                revision = reg_entry.revision
            if reg_entry.doc_code:
                doc_code = reg_entry.doc_code
            sources.append("layer1_document_register")

        return MatchedMetadata(
            project_id=project_id,
            project_name=project_name,
            sharepoint_url=sharepoint_url,
            building=building,
            phase=phase,
            doc_category=doc_category,
            revision=revision,
            doc_code=doc_code,
            resolution_sources=sources,
        )

    def build_document_metadata(
        self,
        file_path: Union[str, Path],
        relative_file_path: Union[str, Path],
        extraction_method: ExtractionMethod,
        file_hash_sha256: Optional[str] = None,
        is_scanned: bool = False,
        is_hvac_diagram: bool = False,
        extraction_confidence: float = 1.0,
        quality_warnings: Optional[List[str]] = None,
        sharepoint_item_id: Optional[str] = None,
        sharepoint_etag: Optional[str] = None,
        sharepoint_url: Optional[str] = None,
    ) -> DocumentMetadata:
        """Create a validated DocumentMetadata object enriched by the 3-layer matcher."""
        actual_path = Path(file_path)
        rel_path = Path(relative_file_path)
        rel_folder = (
            rel_path.parent.as_posix() if rel_path.parent != Path(".") else ""
        )

        if not file_hash_sha256:
            if actual_path.exists():
                file_hash_sha256 = hashlib.sha256(actual_path.read_bytes()).hexdigest()
            else:
                file_hash_sha256 = hashlib.sha256(rel_path.as_posix().encode("utf-8")).hexdigest()

        doc_id = compute_doc_id(rel_path)
        matched = self.match(rel_path)

        return DocumentMetadata(
            doc_id=doc_id,
            file_name=rel_path.name,
            file_extension=rel_path.suffix.lower(),
            file_hash_sha256=file_hash_sha256,
            sharepoint_url=sharepoint_url or matched.sharepoint_url,
            relative_folder_path=rel_folder,
            sharepoint_item_id=sharepoint_item_id,
            sharepoint_etag=sharepoint_etag,
            project_id=matched.project_id,
            project_name=matched.project_name,
            building=matched.building,
            phase=matched.phase,
            doc_category=matched.doc_category,
            extraction_method=extraction_method,
            is_scanned=is_scanned,
            is_hvac_diagram=is_hvac_diagram,
            extraction_confidence=extraction_confidence,
            quality_warnings=list(quality_warnings or []),
        )
