"""Layer 1 (Excel Document Register) and Layer 2 (Master Project Catalog) lookup parsers."""

import csv
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from config import get_settings
from domain import DOC_CATEGORY_RULES, DOCUMENT_REGISTER_FILENAME_RE, DocCategory


class ProjectCatalogEntry(BaseModel):
    """Single row from the master project catalog (CSV/XLSX)."""

    project_id: str
    project_name: str | None = Field(default_factory=lambda: get_settings().default_project_name)
    folder_name: str | None = None
    sharepoint_url: str | None = None
    building: str | None = None
    phase: str | None = None
    client: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentRegisterEntry(BaseModel):
    """Single row parsed from an Excel/CSV Document Register (Documentenlijst / Borderel)."""

    doc_code: str | None = None
    file_name: str | None = None
    title: str | None = None
    building: str | None = None
    phase: str | None = None
    revision: str | None = None
    discipline: str | None = None
    doc_category: DocCategory | None = None
    source_register_path: str | None = None


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _clean_cell(val: Any) -> str:
    if val is None:
        return ""
    text = str(val).strip()
    return "" if text.lower() in {"nan", "none", "null", "n/a", "-"} else text


def _pick(row_map: dict[str, str], keys: set[str]) -> str | None:
    for k in keys:
        if row_map.get(k):
            return row_map[k]
    return None


def _read_tabular_grids(path: Path) -> list[list[list[str]]]:
    """Read a CSV or Excel file into one or more non-empty 2D string grids."""
    ext = path.suffix.lower()
    if ext == ".csv":
        content = None
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                content = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        lines = [line for line in (content or "").splitlines() if line.strip()]
        if not lines:
            return []
        delim = ";" if lines[0].count(";") > lines[0].count(",") else ","
        return [[[_clean_cell(c) for c in row] for row in csv.reader(lines, delimiter=delim)]]

    if ext == ".xlsx":
        import openpyxl

        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        try:
            grids: list[list[list[str]]] = []
            for sheet_name in wb.sheetnames:
                rows = [
                    cleaned
                    for row in wb[sheet_name].iter_rows(values_only=True)
                    if any(cleaned := [_clean_cell(c) for c in row])
                ]
                if rows:
                    grids.append(rows)
            return grids
        finally:
            wb.close()

    raise ValueError(f"Unsupported tabular file extension: {ext}")


def is_document_register_file(file_path: str | Path) -> bool:
    """Return True if the filename matches Flemish document register patterns."""
    path = Path(file_path)
    return path.suffix.lower() in {".xlsx", ".xls", ".csv"} and bool(
        DOCUMENT_REGISTER_FILENAME_RE.search(path.stem)
    )


class ProjectCatalog:
    """Master project catalog mapping SharePoint folders/links to canonical project IDs."""

    PROJECT_ID_COLS = {
        "project_id",
        "projectnummer",
        "project_nr",
        "project_code",
        "projectcode",
        "dossiernummer",
        "dossier_nr",
        "dossier",
        "id",
        "code",
    }
    PROJECT_NAME_COLS = {
        "project_name",
        "projectnaam",
        "naam",
        "project",
        "omschrijving",
        "title",
        "titel",
    }
    FOLDER_NAME_COLS = {
        "folder_name",
        "folder",
        "map",
        "mapnaam",
        "sharepoint_folder",
        "root_folder",
        "directory",
    }
    SHAREPOINT_URL_COLS = {
        "sharepoint_url",
        "sharepoint_link",
        "sharepoint",
        "url",
        "link",
        "folder_url",
    }
    BUILDING_COLS = {"building", "gebouw", "blok", "zone", "vleugel", "locatie"}
    PHASE_COLS = {"phase", "fase", "projectfase", "stadium"}
    CLIENT_COLS = {"client", "klant", "opdrachtgever", "bouwheer"}

    def __init__(self, entries: Iterable[ProjectCatalogEntry] | None = None) -> None:
        self.entries: list[ProjectCatalogEntry] = list(entries or [])
        self._by_id: dict[str, ProjectCatalogEntry] = {}
        self._by_folder: dict[str, ProjectCatalogEntry] = {}
        for entry in self.entries:
            if norm_id := _normalize_token(entry.project_id):
                self._by_id[norm_id] = entry
            if entry.folder_name and (norm_folder := _normalize_token(entry.folder_name)):
                self._by_folder[norm_folder] = entry
            if entry.sharepoint_url:
                slug = entry.sharepoint_url.split("?")[0].rstrip("/").split("/")[-1]
                if norm_slug := _normalize_token(slug.replace("%20", "_")):
                    self._by_folder[norm_slug] = entry

    @classmethod
    def from_file(cls, catalog_path: str | Path) -> "ProjectCatalog":
        path = Path(catalog_path)
        if not path.exists():
            raise FileNotFoundError(f"Catalog file not found: {path}")
        grids = _read_tabular_grids(path)
        if not grids or len(grids[0]) < 2:
            return cls([])

        default_name = get_settings().default_project_name
        known_cols = (
            cls.PROJECT_ID_COLS
            | cls.PROJECT_NAME_COLS
            | cls.FOLDER_NAME_COLS
            | cls.SHAREPOINT_URL_COLS
            | cls.BUILDING_COLS
            | cls.PHASE_COLS
            | cls.CLIENT_COLS
        )
        raw_headers = grids[0][0]
        norm_headers = [_normalize_token(h) for h in raw_headers]
        entries: list[ProjectCatalogEntry] = []
        for row in grids[0][1:]:
            row_map = {
                norm_headers[i]: val
                for i, val in enumerate(row)
                if i < len(norm_headers) and norm_headers[i] and val
            }
            if not row_map:
                continue
            folder_name = _pick(row_map, cls.FOLDER_NAME_COLS)
            project_id = _pick(row_map, cls.PROJECT_ID_COLS) or (
                folder_name.split("/")[0] if folder_name else None
            )
            if not project_id:
                continue
            entries.append(
                ProjectCatalogEntry(
                    project_id=project_id,
                    project_name=_pick(row_map, cls.PROJECT_NAME_COLS) or default_name,
                    folder_name=folder_name,
                    sharepoint_url=_pick(row_map, cls.SHAREPOINT_URL_COLS),
                    building=_pick(row_map, cls.BUILDING_COLS),
                    phase=_pick(row_map, cls.PHASE_COLS),
                    client=_pick(row_map, cls.CLIENT_COLS),
                    extra={
                        raw_headers[i]: row[i]
                        for i in range(min(len(raw_headers), len(row)))
                        if norm_headers[i] and norm_headers[i] not in known_cols and row[i]
                    },
                )
            )
        return cls(entries)

    def match_path(self, relative_path: str | Path) -> ProjectCatalogEntry | None:
        if not self.entries:
            return None
        for part in (p for p in Path(relative_path).parts if p not in {".", "/", "\\"}):
            norm_part = _normalize_token(part)
            if not norm_part:
                continue
            if norm_part in self._by_folder:
                return self._by_folder[norm_part]
            if norm_part in self._by_id:
                return self._by_id[norm_part]
            for norm_folder, entry in self._by_folder.items():
                if norm_part.startswith(norm_folder) or norm_folder.startswith(norm_part):
                    return entry
            for norm_id, entry in self._by_id.items():
                if re.search(rf"(?:^|_){re.escape(norm_id)}(?:_|$)", norm_part):
                    return entry
        return None


class DocumentRegisterIndex:
    """Parses and indexes Excel/CSV Document Registers (Documentenlijsten / Borderellen)."""

    FILENAME_COLS = {"file_name", "filename", "bestandsnaam", "bestand", "file", "document_file"}
    DOC_CODE_COLS = {
        "doc_code",
        "documentnummer",
        "doc_nr",
        "docnr",
        "nummer",
        "code",
        "tekeningnummer",
        "plannummer",
        "plan_nr",
        "referentie",
        "ref",
        "id",
    }
    TITLE_COLS = {
        "title",
        "titel",
        "omschrijving",
        "description",
        "documentnaam",
        "onderwerp",
        "naam",
    }
    BUILDING_COLS = {
        "building",
        "gebouw",
        "blok",
        "vleugel",
        "zone",
        "locatie",
        "bouwdeel",
        "verdieping",
    }
    PHASE_COLS = {"phase", "fase", "projectfase", "stadium", "status"}
    REVISION_COLS = {"revision", "revisie", "rev", "versie", "index", "uitgave"}
    DISCIPLINE_COLS = {"discipline", "lot", "techniek", "installatie", "vakgebied"}
    CATEGORY_COLS = {"doc_category", "category", "categorie", "type", "documenttype", "soort"}

    def __init__(self, entries: Iterable[DocumentRegisterEntry] | None = None) -> None:
        self.entries: list[DocumentRegisterEntry] = []
        self._by_filename: dict[str, DocumentRegisterEntry] = {}
        self._by_stem: dict[str, DocumentRegisterEntry] = {}
        self._by_doc_code: dict[str, DocumentRegisterEntry] = {}
        if entries:
            self.add_entries(entries)

    def add_entries(self, entries: Iterable[DocumentRegisterEntry]) -> None:
        for entry in entries:
            self.entries.append(entry)
            if entry.file_name:
                self._by_filename[Path(entry.file_name).name.lower()] = entry
                if stem_norm := _normalize_token(Path(entry.file_name).stem):
                    self._by_stem[stem_norm] = entry
            if entry.doc_code and (code_norm := _normalize_token(entry.doc_code)):
                self._by_doc_code[code_norm] = entry

    def load_register_file(self, register_path: str | Path) -> list[DocumentRegisterEntry]:
        path = Path(register_path)
        if not path.exists():
            raise FileNotFoundError(f"Document register not found: {path}")
        if path.suffix.lower() not in {".csv", ".xlsx"}:
            return []
        parsed: list[DocumentRegisterEntry] = []
        for grid in _read_tabular_grids(path):
            parsed.extend(self._parse_grid(grid, str(path)))
        self.add_entries(parsed)
        return parsed

    def discover_and_load(self, root_dir: str | Path) -> int:
        root = Path(root_dir)
        if not root.exists():
            return 0
        count = 0
        for file_path in sorted(root.rglob("*")):
            if file_path.is_file() and is_document_register_file(file_path):
                try:
                    count += len(self.load_register_file(file_path))
                except Exception:
                    continue
        return count

    @classmethod
    def _parse_grid(cls, grid: list[list[str]], source_path: str) -> list[DocumentRegisterEntry]:
        if len(grid) < 2:
            return []
        all_known = (
            cls.FILENAME_COLS
            | cls.DOC_CODE_COLS
            | cls.TITLE_COLS
            | cls.BUILDING_COLS
            | cls.PHASE_COLS
            | cls.REVISION_COLS
            | cls.DISCIPLINE_COLS
            | cls.CATEGORY_COLS
        )
        header_idx, best_matches = 0, 0
        for idx, row in enumerate(grid[:15]):
            matches = len({_normalize_token(c) for c in row if c} & all_known)
            if matches > best_matches:
                best_matches, header_idx = matches, idx
        if best_matches == 0:
            return []

        header_row = [_normalize_token(c) for c in grid[header_idx]]
        entries: list[DocumentRegisterEntry] = []
        for row in grid[header_idx + 1 :]:
            row_map = {
                header_row[i]: val
                for i, val in enumerate(row)
                if i < len(header_row) and header_row[i] and val
            }
            if not row_map:
                continue
            file_name = _pick(row_map, cls.FILENAME_COLS)
            doc_code = _pick(row_map, cls.DOC_CODE_COLS)
            title = _pick(row_map, cls.TITLE_COLS)
            if not (file_name or doc_code or title):
                continue
            raw_cat = _pick(row_map, cls.CATEGORY_COLS)
            entries.append(
                DocumentRegisterEntry(
                    doc_code=doc_code,
                    file_name=file_name,
                    title=title,
                    building=_pick(row_map, cls.BUILDING_COLS),
                    phase=_pick(row_map, cls.PHASE_COLS),
                    revision=_pick(row_map, cls.REVISION_COLS),
                    discipline=_pick(row_map, cls.DISCIPLINE_COLS),
                    doc_category=cls._infer_category(
                        " ".join(filter(None, [raw_cat, title, file_name]))
                    ),
                    source_register_path=source_path,
                )
            )
        return entries

    @staticmethod
    def _infer_category(text: str) -> DocCategory | None:
        if not text:
            return None
        for val in DocCategory:
            if val.value.lower() == text.strip().lower():
                return val
        for pattern, category in DOC_CATEGORY_RULES:
            if pattern.search(text):
                return category
        return None

    def match_file(self, file_path: str | Path) -> DocumentRegisterEntry | None:
        if not self.entries:
            return None
        path = Path(file_path)
        fname_lower = path.name.lower()
        stem_norm = _normalize_token(path.stem)
        if fname_lower in self._by_filename:
            return self._by_filename[fname_lower]
        if stem_norm and stem_norm in self._by_stem:
            return self._by_stem[stem_norm]
        for code_norm, entry in self._by_doc_code.items():
            if len(code_norm) >= 3 and re.search(
                rf"(?:^|_){re.escape(code_norm)}(?:_|$)", stem_norm
            ):
                return entry
        for reg_stem, entry in self._by_stem.items():
            if len(reg_stem) >= 5 and (
                stem_norm.startswith(reg_stem) or reg_stem.startswith(stem_norm)
            ):
                return entry
        return None
