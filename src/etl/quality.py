"""Extraction quality evaluator, OCR-noise callout scorer, and limitations audit logger."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from config import get_settings
from domain import ParsedDoc, utc_now

logger = logging.getLogger(__name__)

_MEANINGFUL_TOKEN_RE = re.compile(
    r"(?:[A-Za-zÀ-ÖØ-öø-ÿ]{3,}|[A-Za-z]{1,4}[-_.]?\d{1,4}[A-Za-z]?|"
    r"\d+(?:[.,]\d+)?\s*(?:m[³3]/h|kW|VA|Pa|V|A|Hz|mm|°C|dB))",
    re.IGNORECASE,
)


def score_callout_readability(text: str) -> float:
    """Score readability of a `diagram_callout` block in `[0.0, 1.0]`.

    Legacy-OCR'd electrical/HVAC schematics often emit dozens of 1-2 character glyph fragments
    (`- f`, `- ~n`, `- I`). Returns the fraction of non-empty lines containing a valid technical
    word, equipment tag, or measurement with >=55% alphanumeric characters.
    """
    lines = [re.sub(r"^\s*[-*•]\s*", "", raw).strip() for raw in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return 0.0

    readable = 0
    for line in lines:
        alnum = sum(ch.isalnum() for ch in line)
        if len(line) < 3 or (alnum / len(line)) < 0.55:
            continue
        if _MEANINGFUL_TOKEN_RE.search(line):
            readable += 1
    return round(readable / len(lines), 3)


class LimitationRecord(BaseModel):
    """Structured audit log entry persisted to `limitations_report.jsonl`."""

    logged_at: str = Field(default_factory=lambda: utc_now().isoformat())
    doc_id: str
    file_name: str
    source_path: str
    project: str | None = None
    building: str | None = None
    phase: str | None = None
    doc_category: str = "unknown"
    extraction_method: str = "unknown"
    is_scanned: bool = False
    is_hvac_diagram: bool = False
    needs_ocr: bool = False
    extraction_confidence: float = 0.0
    total_sections: int = 0
    total_char_count: int = 0
    quality_warnings: list[str] = Field(default_factory=list)
    limitation_type: str
    recommended_action: str


class QualityLogger:
    """Evaluates extracted `ParsedDoc` quality and appends limitations to JSONL."""

    def __init__(
        self,
        report_path: str | Path | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self.report_path = (
            Path(report_path) if report_path is not None else settings.limitations_report_path
        )
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.low_confidence_threshold
        )

    def evaluate_and_log(self, doc: ParsedDoc) -> LimitationRecord | None:
        """Inspect `doc` quality; flag `needs_ocr` and log a `LimitationRecord` when below threshold."""
        meta = doc.metadata
        extra = meta.extra
        total_chars = sum(len(s.text.strip()) for s in doc.sections)
        warnings: list[str] = list(extra.get("quality_warnings") or [])
        confidence = float(extra.get("extraction_confidence", 1.0))
        is_scanned = bool(extra.get("is_scanned", False))
        is_hvac_diagram = bool(extra.get("is_hvac_diagram", False))
        needs_ocr = bool(extra.get("needs_ocr", False))

        if len(doc.sections) == 0 or total_chars == 0:
            confidence = min(confidence, 0.2)
            extra["extraction_confidence"] = confidence
            if is_scanned:
                needs_ocr = True
                extra["needs_ocr"] = True

        if confidence >= self.confidence_threshold and not needs_ocr:
            return None

        if (len(doc.sections) == 0 or total_chars == 0) and is_scanned:
            limitation_type = "scanned_pdf_needs_ocr"
            recommended_action = (
                "Scanned PDF produced 0 extracted sections; run Docling/Tesseract or external "
                "high-DPI OCR before re-ingestion."
            )
            msg = "Scanned PDF with 0 extracted text sections (needs_ocr=True)."
            logger.warning(
                "Scanned PDF '%s' (doc_id=%s) yielded 0 sections; flagged needs_ocr=True.",
                meta.source_path,
                doc.document_id,
            )
        elif len(doc.sections) == 0 or total_chars == 0:
            limitation_type = "empty_or_unreadable_document"
            recommended_action = (
                "Verify source file integrity or run external OCR before re-ingestion."
            )
            msg = "No extractable text or table sections found."
        elif is_hvac_diagram:
            limitation_type = "complex_hvac_diagram"
            recommended_action = (
                "Graphical HVAC/PID schematic with sparse text callouts; "
                "cross-reference with equipment schedule (Toestellenlijst) or native CAD (.dwg)."
            )
            msg = "Oversized/vector HVAC diagram with sparse text callouts."
        elif is_scanned or needs_ocr:
            limitation_type = "low_confidence_scan"
            recommended_action = "Scanned archival PDF with low OCR confidence; re-scan at 300+ DPI or review manually."
            msg = "Scanned PDF with low OCR readability confidence."
        else:
            limitation_type = "low_extraction_confidence"
            recommended_action = "Inspect extracted Markdown for layout or encoding artifacts."
            msg = "Extraction confidence below threshold."

        if msg not in warnings:
            warnings.append(msg)
        extra["quality_warnings"] = warnings

        record = LimitationRecord(
            doc_id=doc.document_id,
            file_name=Path(meta.source_path).name,
            source_path=meta.source_path,
            project=meta.project,
            building=meta.building,
            phase=meta.phase,
            doc_category=str(extra.get("doc_category", "unknown")),
            extraction_method=str(extra.get("extraction_method", "unknown")),
            is_scanned=is_scanned,
            is_hvac_diagram=is_hvac_diagram,
            needs_ocr=needs_ocr,
            extraction_confidence=round(confidence, 3),
            total_sections=len(doc.sections),
            total_char_count=total_chars,
            quality_warnings=warnings,
            limitation_type=limitation_type,
            recommended_action=recommended_action,
        )
        self.write_record(record)
        return record

    def log_corrupted_file(
        self,
        relative_path: str | Path,
        error_message: str,
        project: str | None = None,
        building: str | None = None,
        phase: str | None = None,
    ) -> LimitationRecord:
        rel = Path(relative_path)
        record = LimitationRecord(
            doc_id=f"corrupted_{rel.name}",
            file_name=rel.name,
            source_path=rel.as_posix(),
            project=project,
            building=building,
            phase=phase,
            doc_category="unknown",
            extraction_method="failed",
            extraction_confidence=0.0,
            quality_warnings=[f"Extraction error: {error_message}"],
            limitation_type="corrupted_or_unreadable_file",
            recommended_action="Replace corrupted file from SharePoint archive.",
        )
        self.write_record(record)
        return record

    def write_record(self, record: LimitationRecord) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")

    def read_records(self) -> list[LimitationRecord]:
        if not self.report_path.exists():
            return []
        records: list[LimitationRecord] = []
        for line in self.report_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data: dict[str, Any] = json.loads(line)
                records.append(LimitationRecord.model_validate(data))
            except Exception:
                continue
        return records
