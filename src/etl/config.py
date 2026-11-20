"""Configuration constants, paths, OCR thresholds, and Flemish engineering regex patterns."""

import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Pattern, Tuple

from etl.schemas import DocCategory

# ---------------------------------------------------------------------------
# Directory & File Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CREDENTIALS_FILE_PATH: Path = PROJECT_ROOT / "credentials.txt"
ENV_FILE_PATH: Path = PROJECT_ROOT / ".env"

_CREDENTIAL_KEY_ALIASES: Dict[str, str] = {
    "TENANT_ID": "AZURE_TENANT_ID",
    "CLIENT_ID": "AZURE_CLIENT_ID",
    "CLIENT_SECRET": "AZURE_CLIENT_SECRET",
}


def load_credentials_file(
    credentials_path: Optional[Path] = None,
    override: bool = False,
) -> Dict[str, str]:
    """Load key=value pairs from the central `credentials.txt` (and optional `.env`) file."""
    candidate_paths: List[Path] = []
    if credentials_path is not None:
        candidate_paths.append(Path(credentials_path))
    else:
        candidate_paths.extend([CREDENTIALS_FILE_PATH, ENV_FILE_PATH])

    loaded: Dict[str, str] = {}
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                raw_key, _, raw_val = line.partition("=")
                key = raw_key.strip()
                val = raw_val.strip().strip("'").strip('"')
                if not key or not val or val.upper().startswith("YOUR_"):
                    continue
                canonical_key = _CREDENTIAL_KEY_ALIASES.get(key.upper(), key)
                loaded[canonical_key] = val
                if override or canonical_key not in os.environ:
                    os.environ[canonical_key] = val
        except Exception:
            pass
    return loaded


load_credentials_file()

DATA_DIR: Path = PROJECT_ROOT / "data"
CATALOG_DIR: Path = DATA_DIR / "catalog"
DEFAULT_CATALOG_PATH: Path = CATALOG_DIR / "projects_catalog.csv"
RAW_SHAREPOINT_DIR: Path = DATA_DIR / "raw_sharepoint"

PROCESSED_DIR: Path = DATA_DIR / "processed"
MARKDOWN_OUT_DIR: Path = PROCESSED_DIR / "markdown"
JSON_OUT_DIR: Path = PROCESSED_DIR / "json"
MANIFEST_PATH: Path = PROCESSED_DIR / "manifest.json"

LOGS_DIR: Path = DATA_DIR / "logs"
LIMITATIONS_REPORT_PATH: Path = LOGS_DIR / "limitations_report.jsonl"

# ---------------------------------------------------------------------------
# Microsoft Entra ID (Azure AD) & SharePoint Graph API Configuration
# ---------------------------------------------------------------------------
AZURE_TENANT_ID: str = os.environ.get("AZURE_TENANT_ID", "")
AZURE_CLIENT_ID: str = os.environ.get("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET: str = os.environ.get("AZURE_CLIENT_SECRET", "")
TEMP_BEARER_TOKEN: str = os.environ.get("TEMP_BEARER_TOKEN", "")
SHAREPOINT_HOSTNAME: str = os.environ.get("SHAREPOINT_HOSTNAME", "")
SHAREPOINT_SITE_PATH: str = os.environ.get("SHAREPOINT_SITE_PATH", "")
SHAREPOINT_DRIVE_NAME: str = os.environ.get("SHAREPOINT_DRIVE_NAME", "Documents")
GRAPH_API_BASE_URL: str = "https://graph.microsoft.com/v1.0"
GRAPH_DEFAULT_SCOPE: str = "https://graph.microsoft.com/.default"

# ---------------------------------------------------------------------------
# Default Domain Metadata Values (None = unresolved; never emit placeholder strings)
# ---------------------------------------------------------------------------
DEFAULT_PROJECT_ID: Optional[str] = None
DEFAULT_PROJECT_NAME: str = "AZ Sint-Jan Brugge"
DEFAULT_BUILDING: Optional[str] = None
DEFAULT_PHASE: Optional[str] = None

# ---------------------------------------------------------------------------
# Supported Extensions
# ---------------------------------------------------------------------------
SUPPORTED_EXTRACTION_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls"}

# ---------------------------------------------------------------------------
# PDF Inspection & Quality Thresholds
# ---------------------------------------------------------------------------
MIN_DIGITAL_TEXT_CHARS_PER_PAGE: int = 50
# Standard A3 is ~842 x 1191 pt. Pages larger than 1250 pt on their longest edge
# (such as A2, A1, A0 architectural/HVAC plans) are flagged as oversized drawings.
OVERSIZED_PAGE_MIN_DIMENSION_PT: float = 1250.0
HIGH_VECTOR_DRAWINGS_THRESHOLD: int = 250
LOW_CONFIDENCE_THRESHOLD: float = 0.65
SPARSE_DIAGRAM_CHAR_THRESHOLD: int = 150

# ---------------------------------------------------------------------------
# Document Register Detection (Layer 1)
# ---------------------------------------------------------------------------
DOCUMENT_REGISTER_FILENAME_RE: Pattern[str] = re.compile(
    r"(documentenlijst|doc(?:ument)?[_\-\s]*lijst|borderel|document[_\-\s]*register|"
    r"planlijst|plannenlijst|index(?:[_\-\s]*lijst)?|overzicht(?:slijst)?)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Project ID Extraction from Folder/File Names (Fallback when catalog missing)
# ---------------------------------------------------------------------------
PROJECT_ID_PATTERNS: List[Pattern[str]] = [
    # Matches PRJ-2021-04, PRJ_104, PRJ-104, PROJ_2023_012, ING-2022-089 even when followed by _AZ_Sint_Jan
    re.compile(
        r"(?:^|[_\-\s/\\])((?:PRJ|PROJ|ING)[_\-]\d{2,4}(?:[_\-]\d{1,4})?)(?=$|[_\-\s/\\.])",
        re.IGNORECASE,
    ),
    # Ingenium dossier numbers: `06038.004`, `6038.004`, `06038-004`, `06038_004`.
    # Deliberately rejects supplier order numbers (`17-341-1883`), drawing/document codes
    # (`1003-004`, `9180.A318.01`) and external references (`042193.21.002`).
    re.compile(r"(?<![\d.])(0\d{4}[._\-]\d{3}|\d{4,5}\.\d{3})(?![\d]|[._\-]\d)"),
]

# Ingenium SharePoint sites are created per dossier, e.g. `/sites/06038004-INT/...`.
SHAREPOINT_SITE_SLUG_RE: Pattern[str] = re.compile(r"/sites/([^/?#]+)", re.IGNORECASE)
SHAREPOINT_SITE_PROJECT_RE: Pattern[str] = re.compile(
    r"^(\d{4,5})[._\-]?(\d{3})(?:$|[-_][A-Za-z0-9]+$)"
)

# ---------------------------------------------------------------------------
# Phase Detection Rules (Layer 3)
# ---------------------------------------------------------------------------
# Each rule: (full-name pattern, abbreviation pattern, canonical label).
# Two-letter abbreviations (VO/DO/AO/UO/AB) are only trusted in folder/file names: inside
# drawing text they collide with BMS/electrical codes (DO = digital output, AO = analog output).
_PB = r"(?:^|[_\-\s/\\0-9])"  # left boundary
_PA = r"(?:$|[_\-\s/\\.,;?!])"  # right boundary
PHASE_RULES: List[Tuple[Pattern[str], Pattern[str], str]] = [
    (
        re.compile(_PB + r"(?:voorontwerp|preliminary(?:\s*design)?)" + _PA, re.IGNORECASE),
        re.compile(_PB + r"VO" + _PA, re.IGNORECASE),
        "Voorontwerp (Preliminary Design)",
    ),
    (
        re.compile(_PB + r"(?:definitief[_\-\s]*ontwerp|final[_\-\s]*design)" + _PA, re.IGNORECASE),
        re.compile(_PB + r"DO" + _PA, re.IGNORECASE),
        "Definitief Ontwerp (Final Design)",
    ),
    (
        re.compile(_PB + r"(?:aanbesteding|bestek(?:fase)?|tender)" + _PA, re.IGNORECASE),
        re.compile(_PB + r"AO" + _PA, re.IGNORECASE),
        "Aanbesteding (Tender / Specifications)",
    ),
    (
        re.compile(_PB + r"(?:uitvoering(?:sdossier)?|execution)" + _PA, re.IGNORECASE),
        re.compile(_PB + r"UO" + _PA, re.IGNORECASE),
        "Uitvoering (Execution)",
    ),
    (
        re.compile(_PB + r"(?:as[_\-\s]*built|oplevering(?:sdossier)?)" + _PA, re.IGNORECASE),
        re.compile(_PB + r"AB" + _PA, re.IGNORECASE),
        "As-Built",
    ),
]

# ---------------------------------------------------------------------------
# Building / Wing / Floor / Zone Detection Rules (Layer 3)
# ---------------------------------------------------------------------------
# Block/wing/building codes are short (A, B2, L, 12) or compass names. Longer words that follow
# "Blok" in drawing title blocks (e.g. "BLOK | PEIL", "Blok nr") are not building identifiers.
_BUILDING_CODE = r"(Noord|Zuid|Oost|West|[A-Z]{1,2}\d{0,2}|\d{1,3})"
BUILDING_LABEL_STOPWORDS = frozenset({"NR", "NO", "DE", "EN", "IN", "OP", "TE", "VAN", "PE"})

# Structural identifiers (Blok, Vleugel, Gebouw, Campus, Verdieping, Niv)
STRUCTURAL_BUILDING_PATTERNS: List[Tuple[Pattern[str], str]] = [
    (
        re.compile(
            r"(?:^|[_\-\s/\\])(Blok)[_\-\s]+" + _BUILDING_CODE + r"(?=$|[_\-\s/\\.,;)])",
            re.IGNORECASE,
        ),
        "Blok {1}",
    ),
    (
        re.compile(
            r"(?:^|[_\-\s/\\])(Vleugel)[_\-\s]+" + _BUILDING_CODE + r"(?=$|[_\-\s/\\.,;)])",
            re.IGNORECASE,
        ),
        "Vleugel {1}",
    ),
    (
        re.compile(
            r"(?:^|[_\-\s/\\])(Gebouw)[_\-\s]+([A-Z]\d{0,2}|\d{1,3})(?=$|[_\-\s/\\.,;)])",
            re.IGNORECASE,
        ),
        "Gebouw {1}",
    ),
    (
        re.compile(
            r"(?:^|[_\-\s/\\])(Campus)[_\-\s]+(Sint[_\-\s]*Jan|SFX|Henri[_\-\s]*Serruys|[A-Za-z0-9]+)(?=$|[_\-\s/\\.])",
            re.IGNORECASE,
        ),
        "Campus {1}",
    ),
    (
        re.compile(
            r"(?:^|[_\-\s/\\])(Verdieping)[_\s]*([+-]?\d{1,2})(?=$|[_\-\s/\\.])",
            re.IGNORECASE,
        ),
        "Verdieping {1}",
    ),
    (
        re.compile(
            r"(?:^|[_\-\s/\\])Niv(?:eau)?\.?[_\s]*([+-]?\d{1,2})(?=$|[_\-\s/\\.])",
            re.IGNORECASE,
        ),
        "Niv {0}",
    ),
]

# Functional hospital zones at AZ Sint-Jan Brugge
HOSPITAL_ZONE_PATTERNS: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"(?:^|[_\-\s/\\])operatiekwartier(?=$|[_\-\s/\\.])", re.IGNORECASE), "Operatiekwartier"),
    (re.compile(r"(?:^|[_\-\s/\\])spoed(?:afdeling|eisende)?(?=$|[_\-\s/\\.])", re.IGNORECASE), "Spoed"),
    (re.compile(r"(?:^|[_\-\s/\\])labo(?:ratorium)?(?=$|[_\-\s/\\.])", re.IGNORECASE), "Labo"),
    (re.compile(r"(?:^|[_\-\s/\\])technische[_\-\s]+ruimte(?=$|[_\-\s/\\.])", re.IGNORECASE), "Technische Ruimte"),
    (re.compile(r"(?:^|[_\-\s/\\])intensieve[_\-\s]+zorgen(?=$|[_\-\s/\\.])", re.IGNORECASE), "Intensieve Zorgen"),
    (re.compile(r"(?:^|[_\-\s/\\])radiologie(?=$|[_\-\s/\\.])", re.IGNORECASE), "Radiologie"),
    (re.compile(r"(?:^|[_\-\s/\\])apotheek(?=$|[_\-\s/\\.])", re.IGNORECASE), "Apotheek"),
    (re.compile(r"(?:^|[_\-\s/\\])centrale[_\-\s]+sterilisatie(?=$|[_\-\s/\\.])", re.IGNORECASE), "Centrale Sterilisatie"),
    (re.compile(r"(?:^|[_\-\s/\\])materniteit(?=$|[_\-\s/\\.])", re.IGNORECASE), "Materniteit"),
    (re.compile(r"(?:^|[_\-\s/\\])pediatrie(?=$|[_\-\s/\\.])", re.IGNORECASE), "Pediatrie"),
    (re.compile(r"(?:^|[_\-\s/\\])polikliniek(?=$|[_\-\s/\\.])", re.IGNORECASE), "Polikliniek"),
]

# ---------------------------------------------------------------------------
# Document Category Rules (Layer 3)
# ---------------------------------------------------------------------------
# Evaluated in priority order: explicit document type keywords (e.g., bestek,
# documentenlijst, berekeningsnota, meetstaat, principeschema, verslag) take
# precedence over discipline/topic keywords (e.g., ventilatie, luchtgroep).
PRIMARY_DOC_CATEGORY_RULES: List[Tuple[Pattern[str], DocCategory]] = [
    (DOCUMENT_REGISTER_FILENAME_RE, DocCategory.DOCUMENT_REGISTER),
    (
        re.compile(r"(bestek|lastenboek|technische[_\-\s]*specificatie)", re.IGNORECASE),
        DocCategory.SPECIFICATION,
    ),
    (
        re.compile(
            r"(berekeningsnota|berekening|calculatie|warmteverlies|koellast|dimensionering)",
            re.IGNORECASE,
        ),
        DocCategory.CALCULATION,
    ),
    (
        re.compile(
            r"(meetstaat|toestellenlijst|materiaallijst|vorderingsstaat|equipment[_\-\s]*schedule)",
            re.IGNORECASE,
        ),
        DocCategory.SCHEDULE_TABLE,
    ),
    (
        re.compile(
            r"(principeschema|schema|grondplan|plan|pid|p&id|isometrie|tekening|diagram)",
            re.IGNORECASE,
        ),
        DocCategory.DRAWING_DIAGRAM,
    ),
    (
        re.compile(
            r"(werfverslag|verslag|notulen|vergadering|rapport|audit|opname)",
            re.IGNORECASE,
        ),
        DocCategory.REPORT_MEETING,
    ),
]

SECONDARY_DOC_CATEGORY_RULES: List[Tuple[Pattern[str], DocCategory]] = [
    (
        re.compile(r"(ventilatie|luchtgroep(?:en)?|debiet(?:en)?|roosterlijst)", re.IGNORECASE),
        DocCategory.SCHEDULE_TABLE,
    ),
]
