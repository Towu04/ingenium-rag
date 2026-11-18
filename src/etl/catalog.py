"""Google Sheet Project Catalog and Excel Document Register parsers.

Implements Layer 1 (Excel Document Register / Documentenlijst lookup) and
Layer 2 (Master Google Sheet Project Catalog lookup) for metadata matching.
"""

import csv
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from etl.config import (
    DEFAULT_PROJECT_NAME,
    DOCUMENT_REGISTER_FILENAME_RE,
    PRIMARY_DOC_CATEGORY_RULES,
    SECONDARY_DOC_CATEGORY_RULES,
)
from etl.schemas import DocCategory, DocumentRegisterEntry, ProjectCatalogEntry


def _normalize_token(text: str) -> str:
    """Normalize text to lowercase alphanumeric tokens separated by single underscores."""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _clean_cell(val: Any) -> str:
    """Convert a cell value to a stripped string, returning empty string for None/NaN."""
    if val is None:
        return ""
    text = str(val).strip()
    if text.lower() in {"nan", "none", "null", "n/a", "-"}:
        return ""
    return text


def is_document_register_file(file_path: Union[str, Path]) -> bool:
    """Return True if the filename matches Flemish document register patterns."""
    path = Path(file_path)
    if path.suffix.lower() not in {".xlsx", ".xls", ".csv"}:
        return False
    return bool(DOCUMENT_REGISTER_FILENAME_RE.search(path.stem))


class ProjectCatalog:
    """Master Google Sheet Project Catalog mapping SharePoint folders/links to projects."""

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

    def __init__(self, entries: Optional[Iterable[ProjectCatalogEntry]] = None) -> None:
        self.entries: List[ProjectCatalogEntry] = list(entries or [])
        self._by_id: Dict[str, ProjectCatalogEntry] = {}
        self._by_folder: Dict[str, ProjectCatalogEntry] = {}
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        self._by_id.clear()
        self._by_folder.clear()
        for entry in self.entries:
            norm_id = _normalize_token(entry.project_id)
            if norm_id:
                self._by_id[norm_id] = entry
            if entry.folder_name:
                norm_folder = _normalize_token(entry.folder_name)
                if norm_folder:
                    self._by_folder[norm_folder] = entry
            if entry.sharepoint_url:
                slug = self._extract_url_folder_slug(entry.sharepoint_url)
                if slug:
                    self._by_folder[_normalize_token(slug)] = entry

    @staticmethod
    def _extract_url_folder_slug(url: str) -> Optional[str]:
        """Extract the trailing folder name from a SharePoint URL."""
        cleaned = url.split("?")[0].rstrip("/")
        if not cleaned:
            return None
        last_segment = cleaned.split("/")[-1]
        # URL decode %20 etc.
        last_segment = re.sub(r"%20", "_", last_segment)
        return last_segment or None

    @classmethod
    def from_file(cls, catalog_path: Union[str, Path]) -> "ProjectCatalog":
        """Load project catalog from a Google Sheet export (.csv or .xlsx)."""
        path = Path(catalog_path)
        if not path.exists():
            raise FileNotFoundError(f"Catalog file not found: {path}")

        if path.suffix.lower() == ".csv":
            rows = cls._read_csv_rows(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            rows = cls._read_excel_rows(path)
        else:
            raise ValueError(f"Unsupported catalog file extension: {path.suffix}")

        entries = cls._parse_dict_rows(rows)
        return cls(entries)

    @staticmethod
    def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
        content = None
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                content = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if not content or not content.strip():
            return []

        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            return []

        # Detect delimiter (comma vs semicolon common in Belgian Excel CSV exports)
        first_line = lines[0]
        delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
        reader = csv.DictReader(lines, delimiter=delimiter)
        return [dict(row) for row in reader]

    @staticmethod
    def _read_excel_rows(path: Path) -> List[Dict[str, Any]]:
        import openpyxl

        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        try:
            sheet = wb.active
            if sheet is None:
                return []
            raw_rows = list(sheet.iter_rows(values_only=True))
            if not raw_rows:
                return []
            headers = [_clean_cell(h) or f"col_{idx}" for idx, h in enumerate(raw_rows[0])]
            dict_rows: List[Dict[str, Any]] = []
            for row in raw_rows[1:]:
                if not any(_clean_cell(c) for c in row):
                    continue
                dict_rows.append(dict(zip(headers, row)))
            return dict_rows
        finally:
            wb.close()

    @classmethod
    def _parse_dict_rows(cls, rows: List[Dict[str, Any]]) -> List[ProjectCatalogEntry]:
        entries: List[ProjectCatalogEntry] = []
        for raw_row in rows:
            normalized_map: Dict[str, Tuple[str, str]] = {}
            for k, v in raw_row.items():
                if k is None:
                    continue
                norm_k = _normalize_token(str(k))
                val_str = _clean_cell(v)
                if norm_k and val_str:
                    normalized_map[norm_k] = (str(k), val_str)

            if not normalized_map:
                continue

            project_id = cls._pick_first(normalized_map, cls.PROJECT_ID_COLS)
            project_name = (
                cls._pick_first(normalized_map, cls.PROJECT_NAME_COLS) or DEFAULT_PROJECT_NAME
            )
            folder_name = cls._pick_first(normalized_map, cls.FOLDER_NAME_COLS)
            sharepoint_url = cls._pick_first(normalized_map, cls.SHAREPOINT_URL_COLS)
            building = cls._pick_first(normalized_map, cls.BUILDING_COLS)
            phase = cls._pick_first(normalized_map, cls.PHASE_COLS)
            client = cls._pick_first(normalized_map, cls.CLIENT_COLS) or DEFAULT_PROJECT_NAME

            # If project_id is missing but folder_name is present, derive project_id from folder_name
            if not project_id and folder_name:
                project_id = folder_name.split("/")[0]
            if not project_id:
                continue

            known_cols = (
                cls.PROJECT_ID_COLS
                | cls.PROJECT_NAME_COLS
                | cls.FOLDER_NAME_COLS
                | cls.SHAREPOINT_URL_COLS
                | cls.BUILDING_COLS
                | cls.PHASE_COLS
                | cls.CLIENT_COLS
            )
            extra = {
                orig_k: val
                for norm_k, (orig_k, val) in normalized_map.items()
                if norm_k not in known_cols
            }

            entries.append(
                ProjectCatalogEntry(
                    project_id=project_id,
                    project_name=project_name,
                    folder_name=folder_name,
                    sharepoint_url=sharepoint_url,
                    building=building,
                    phase=phase,
                    client=client,
                    extra=extra,
                )
            )
        return entries

    @staticmethod
    def _pick_first(
        normalized_map: Dict[str, Tuple[str, str]], candidate_keys: set[str]
    ) -> Optional[str]:
        for key in candidate_keys:
            if key in normalized_map:
                return normalized_map[key][1]
        return None

    def match_path(self, relative_path: Union[str, Path]) -> Optional[ProjectCatalogEntry]:
        """Match a relative SharePoint path or project folder to a catalog entry."""
        if not self.entries:
            return None

        path_obj = Path(relative_path)
        parts = [p for p in path_obj.parts if p not in {".", "/", "\\"}]
        if not parts:
            return None

        # Check each path segment (starting from top-level project folder)
        for part in parts:
            norm_part = _normalize_token(part)
            if not norm_part:
                continue

            # 1. Exact folder name or project ID match
            if norm_part in self._by_folder:
                return self._by_folder[norm_part]
            if norm_part in self._by_id:
                return self._by_id[norm_part]

            # 2. Prefix or token boundary match against project_id or folder_name
            for norm_folder, entry in self._by_folder.items():
                if norm_part.startswith(norm_folder) or norm_folder.startswith(norm_part):
                    return entry

            for norm_id, entry in self._by_id.items():
                # Match e.g. 'prj_104' inside 'prj_104_az_sint_jan'
                if re.search(rf"(?:^|_){re.escape(norm_id)}(?:_|$)", norm_part):
                    return entry

        return None


class DocumentRegisterIndex:
    """Parses and indexes Excel/CSV Document Registers (Documentenlijsten / Borderellen)."""

    FILENAME_COLS = {
        "file_name",
        "filename",
        "bestandsnaam",
        "bestand",
        "file",
        "document_file",
    }
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

    def __init__(self, entries: Optional[Iterable[DocumentRegisterEntry]] = None) -> None:
        self.entries: List[DocumentRegisterEntry] = []
        self._by_filename: Dict[str, DocumentRegisterEntry] = {}
        self._by_stem: Dict[str, DocumentRegisterEntry] = {}
        self._by_doc_code: Dict[str, DocumentRegisterEntry] = {}
        if entries:
            self.add_entries(entries)

    def add_entries(self, entries: Iterable[DocumentRegisterEntry]) -> None:
        """Add parsed document register entries to the lookup indices."""
        for entry in entries:
            self.entries.append(entry)
            if entry.file_name:
                fname_lower = Path(entry.file_name).name.lower()
                stem_norm = _normalize_token(Path(entry.file_name).stem)
                self._by_filename[fname_lower] = entry
                if stem_norm:
                    self._by_stem[stem_norm] = entry
            if entry.doc_code:
                code_norm = _normalize_token(entry.doc_code)
                if code_norm:
                    self._by_doc_code[code_norm] = entry

    def load_register_file(self, register_path: Union[str, Path]) -> List[DocumentRegisterEntry]:
        """Parse an Excel (.xlsx) or CSV (.csv) document register and index its rows."""
        path = Path(register_path)
        if not path.exists():
            raise FileNotFoundError(f"Document register not found: {path}")

        if path.suffix.lower() == ".csv":
            raw_grids = [self._read_csv_grid(path)]
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            raw_grids = self._read_excel_grids(path)
        else:
            return []

        parsed_entries: List[DocumentRegisterEntry] = []
        for grid in raw_grids:
            parsed_entries.extend(self._parse_grid(grid, str(path)))

        self.add_entries(parsed_entries)
        return parsed_entries

    def discover_and_load(self, root_dir: Union[str, Path]) -> int:
        """Scan a directory tree for document register files and load all of them."""
        root = Path(root_dir)
        if not root.exists():
            return 0
        count = 0
        for file_path in sorted(root.rglob("*")):
            if file_path.is_file() and is_document_register_file(file_path):
                try:
                    loaded = self.load_register_file(file_path)
                    count += len(loaded)
                except Exception:
                    continue
        return count

    @staticmethod
    def _read_csv_grid(path: Path) -> List[List[str]]:
        content = None
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                content = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if not content:
            return []
        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            return []
        delimiter = ";" if lines[0].count(";") > lines[0].count(",") else ","
        reader = csv.reader(lines, delimiter=delimiter)
        return [[_clean_cell(cell) for cell in row] for row in reader]

    @staticmethod
    def _read_excel_grids(path: Path) -> List[List[List[str]]]:
        import openpyxl

        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        grids: List[List[List[str]]] = []
        try:
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                rows: List[List[str]] = []
                for row in sheet.iter_rows(values_only=True):
                    cleaned = [_clean_cell(c) for c in row]
                    if any(cleaned):
                        rows.append(cleaned)
                if rows:
                    grids.append(rows)
            return grids
        finally:
            wb.close()

    @classmethod
    def _parse_grid(
        cls, grid: List[List[str]], source_path: str
    ) -> List[DocumentRegisterEntry]:
        if len(grid) < 2:
            return []

        all_known_headers = (
            cls.FILENAME_COLS
            | cls.DOC_CODE_COLS
            | cls.TITLE_COLS
            | cls.BUILDING_COLS
            | cls.PHASE_COLS
            | cls.REVISION_COLS
            | cls.DISCIPLINE_COLS
            | cls.CATEGORY_COLS
        )

        # Locate the header row within the first 15 non-empty rows
        header_row_idx = 0
        best_match_count = 0
        for idx, row in enumerate(grid[:15]):
            norm_cells = {_normalize_token(c) for c in row if c}
            matches = len(norm_cells & all_known_headers)
            if matches > best_match_count:
                best_match_count = matches
                header_row_idx = idx

        if best_match_count == 0:
            return []

        header_row = [_normalize_token(c) for c in grid[header_row_idx]]
        entries: List[DocumentRegisterEntry] = []

        for row in grid[header_row_idx + 1 :]:
            row_map: Dict[str, str] = {}
            for col_idx, cell_val in enumerate(row):
                if col_idx < len(header_row) and header_row[col_idx] and cell_val:
                    row_map[header_row[col_idx]] = cell_val

            if not row_map:
                continue

            file_name = cls._pick(row_map, cls.FILENAME_COLS)
            doc_code = cls._pick(row_map, cls.DOC_CODE_COLS)
            title = cls._pick(row_map, cls.TITLE_COLS)
            building = cls._pick(row_map, cls.BUILDING_COLS)
            phase = cls._pick(row_map, cls.PHASE_COLS)
            revision = cls._pick(row_map, cls.REVISION_COLS)
            discipline = cls._pick(row_map, cls.DISCIPLINE_COLS)
            raw_cat = cls._pick(row_map, cls.CATEGORY_COLS)

            if not (file_name or doc_code or title):
                continue

            doc_category = cls._infer_category_from_text(
                " ".join(filter(None, [raw_cat, title, file_name]))
            )

            entries.append(
                DocumentRegisterEntry(
                    doc_code=doc_code,
                    file_name=file_name,
                    title=title,
                    building=building,
                    phase=phase,
                    revision=revision,
                    discipline=discipline,
                    doc_category=doc_category,
                    source_register_path=source_path,
                )
            )
        return entries

    @staticmethod
    def _pick(row_map: Dict[str, str], keys: set[str]) -> Optional[str]:
        for k in keys:
            if k in row_map and row_map[k]:
                return row_map[k]
        return None

    @staticmethod
    def _infer_category_from_text(text: str) -> Optional[DocCategory]:
        if not text:
            return None
        for val in DocCategory:
            if val.value.lower() == text.strip().lower():
                return val
        for pattern, category in PRIMARY_DOC_CATEGORY_RULES:
            if pattern.search(text):
                return category
        for pattern, category in SECONDARY_DOC_CATEGORY_RULES:
            if pattern.search(text):
                return category
        return None

    def match_file(self, file_path: Union[str, Path]) -> Optional[DocumentRegisterEntry]:
        """Look up a file in the loaded Document Registers by filename, stem, or doc_code."""
        if not self.entries:
            return None

        path = Path(file_path)
        fname_lower = path.name.lower()
        stem_norm = _normalize_token(path.stem)

        # 1. Exact filename match
        if fname_lower in self._by_filename:
            return self._by_filename[fname_lower]

        # 2. Exact normalized stem match
        if stem_norm and stem_norm in self._by_stem:
            return self._by_stem[stem_norm]

        # 3. Document code match (e.g. 'HVAC-BLOKB-01' inside 'HVAC-BLOKB-01_Bestek.pdf')
        for code_norm, entry in self._by_doc_code.items():
            if len(code_norm) >= 3 and re.search(
                rf"(?:^|_){re.escape(code_norm)}(?:_|$)", stem_norm
            ):
                return entry

        # 4. Registered stem prefix/substring match inside actual filename
        for reg_stem, entry in self._by_stem.items():
            if len(reg_stem) >= 5 and (
                stem_norm.startswith(reg_stem) or reg_stem.startswith(stem_norm)
            ):
                return entry

        return None
