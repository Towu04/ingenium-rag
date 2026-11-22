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
    """Return the fraction in `[0.0, 1.0]` of non-empty callout lines with readable tokens."""
    lines = [
        cleaned
        for raw in text.splitlines()
        if (cleaned := re.sub(r"^\s*[-*•]\s*", "", raw).strip())
    ]
    if not lines:
        return 0.0
    readable = sum(
        1
        for line in lines
        if len(line) >= 3
        and (sum(ch.isalnum() for ch in line) / len(line)) >= 0.55
        and _MEANINGFUL_TOKEN_RE.search(line)
    )
    return round(readable / len(lines), 3)


def _classify_limitation(
    *, is_empty: bool, is_scanned: bool, is_hvac_diagram: bool, needs_ocr: bool
) -> tuple[str, str, str]:
    if is_empty and is_scanned:
        return (
            "scanned_pdf_needs_ocr",
            "Scanned PDF produced 0 extracted sections; run OCR before re-ingestion.",
            "Scanned PDF with 0 extracted text sections (needs_ocr=True).",
        )
    if is_empty:
        return (
            "empty_or_unreadable_document",
            "Verify source file integrity or run external OCR before re-ingestion.",
            "No extractable text or table sections found.",
        )
    if is_hvac_diagram:
        return (
            "complex_hvac_diagram",
            "Graphical schematic with sparse callouts; cross-reference with equipment schedule or CAD.",
            "Oversized/vector diagram with sparse text callouts.",
        )
    if is_scanned or needs_ocr:
        return (
            "low_confidence_scan",
            "Scanned PDF with low OCR confidence; re-scan at 300+ DPI or review manually.",
            "Scanned PDF with low OCR readability confidence.",
        )
    return (
        "low_extraction_confidence",
        "Inspect extracted Markdown for layout or encoding artifacts.",
        "Extraction confidence below threshold.",
    )


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
        meta, extra = doc.metadata, doc.metadata.extra
        total_chars = sum(len(s.text.strip()) for s in doc.sections)
        is_empty = len(doc.sections) == 0 or total_chars == 0
        confidence = float(extra.get("extraction_confidence", 1.0))
        is_scanned = bool(extra.get("is_scanned", False))
        is_hvac_diagram = bool(extra.get("is_hvac_diagram", False))
        needs_ocr = bool(extra.get("needs_ocr", False))
        truncated_sheets: list[str] = list(extra.get("xlsx_truncated_sheets") or [])

        if is_empty:
            confidence = min(confidence, 0.2)
            extra["extraction_confidence"] = confidence
            if is_scanned:
                needs_ocr = True
                extra["needs_ocr"] = True

        if confidence >= self.confidence_threshold and not needs_ocr and not truncated_sheets:
            return None

        if confidence >= self.confidence_threshold and not needs_ocr and truncated_sheets:
            limitation_type = "xlsx_sheet_truncated"
            action = "Large spreadsheet truncated at row cap; inspect original workbook or summary tab for full data."
            msg = f"Truncated Excel sheet(s): {', '.join(truncated_sheets[:5])}"
        else:
            limitation_type, action, msg = _classify_limitation(
                is_empty=is_empty,
                is_scanned=is_scanned,
                is_hvac_diagram=is_hvac_diagram,
                needs_ocr=needs_ocr,
            )
        if limitation_type == "scanned_pdf_needs_ocr":
            logger.warning(
                "Scanned PDF '%s' (doc_id=%s) yielded 0 sections; flagged needs_ocr=True.",
                meta.source_path,
                doc.document_id,
            )

        warnings: list[str] = list(extra.get("quality_warnings") or [])
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
            recommended_action=action,
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
            extraction_method="failed",
            quality_warnings=[f"Extraction error: {error_message}"],
            limitation_type="corrupted_or_unreadable_file",
            recommended_action="Verify or replace corrupted source file.",
        )
        self.write_record(record)
        return record

    def log_unauthorized(
        self,
        target: str,
        status_code: int,
        reason: str = "",
        *,
        project: str | None = None,
    ) -> LimitationRecord:
        rel = Path(target)
        record = LimitationRecord(
            doc_id=f"unauthorized_{rel.name or 'site'}",
            file_name=rel.name or target,
            source_path=target,
            project=project,
            extraction_method="sharepoint_auth",
            quality_warnings=[f"HTTP {status_code}: {reason or 'Access denied'}"],
            limitation_type="unauthorized_access",
            recommended_action="Grant Microsoft Entra app Sites.Read.All / site permission for this target.",
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
            if not line.strip():
                continue
            try:
                data: dict[str, Any] = json.loads(line)
                records.append(LimitationRecord.model_validate(data))
            except Exception:
                continue
        return records
