"""Extraction quality evaluator and limitations logger (`limitations_report.jsonl`).

Automatically records unreadable scans, corrupted files, or complex CAD/HVAC diagrams
where text extraction is unreliable into a structured JSONL report for client handover.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from etl.config import LIMITATIONS_REPORT_PATH, LOW_CONFIDENCE_THRESHOLD
from etl.schemas import ExtractedDocument


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LimitationRecord(BaseModel):
    """Structured audit log entry written to `data/logs/limitations_report.jsonl`."""

    logged_at: str = Field(default_factory=_utc_now_iso)
    doc_id: str
    file_name: str
    relative_folder_path: str
    project_id: str
    building: str
    phase: str
    doc_category: str
    extraction_method: str
    is_scanned: bool = False
    is_hvac_diagram: bool = False
    extraction_confidence: float
    total_blocks: int = 0
    total_char_count: int = 0
    quality_warnings: List[str] = Field(default_factory=list)
    limitation_type: str = Field(
        ...,
        description="e.g., 'low_confidence_scan', 'complex_hvac_diagram', 'empty_or_corrupted'",
    )
    recommended_action: str


class QualityLogger:
    """Evaluates document extraction quality and appends limitations to JSONL audit log."""

    def __init__(
        self,
        report_path: Union[str, Path] = LIMITATIONS_REPORT_PATH,
        confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.report_path = Path(report_path)
        self.confidence_threshold = confidence_threshold

    def evaluate_and_log(self, doc: ExtractedDocument) -> Optional[LimitationRecord]:
        """Populate document-level quality warnings and log a LimitationRecord if confidence is below threshold."""
        meta = doc.metadata
        total_chars = sum(len(b.markdown_content.strip()) for b in doc.blocks)
        warnings = list(meta.quality_warnings)

        if len(doc.blocks) == 0 or total_chars == 0:
            meta.extraction_confidence = min(meta.extraction_confidence, 0.2)

        if meta.extraction_confidence >= self.confidence_threshold:
            return None

        if len(doc.blocks) == 0 or total_chars == 0:
            limitation_type = "empty_or_unreadable_document"
            recommended_action = (
                "Verify source file integrity or run external high-DPI OCR before re-ingestion."
            )
            msg = "No extractable text or table blocks found."
        elif meta.is_hvac_diagram:
            limitation_type = "complex_hvac_diagram"
            recommended_action = (
                "Graphical HVAC/PID schematic with sparse text callouts; "
                "cross-reference with equipment schedule (Toestellenlijst) or native CAD (.dwg)."
            )
            msg = "Oversized/vector HVAC diagram with sparse text callouts."
        elif meta.is_scanned:
            limitation_type = "low_confidence_scan"
            recommended_action = (
                "Scanned archival PDF with low OCR confidence; "
                "re-scan at 300+ DPI or review OCR output manually."
            )
            msg = "Scanned PDF with low OCR readability confidence."
        else:
            limitation_type = "low_extraction_confidence"
            recommended_action = "Inspect extracted Markdown for layout or encoding artifacts."
            msg = "Extraction confidence below threshold."

        if msg not in warnings:
            warnings.append(msg)
        meta.quality_warnings = warnings

        record = LimitationRecord(
            doc_id=meta.doc_id,
            file_name=meta.file_name,
            relative_folder_path=meta.relative_folder_path,
            project_id=meta.project_id,
            building=meta.building,
            phase=meta.phase,
            doc_category=meta.doc_category.value,
            extraction_method=meta.extraction_method.value,
            is_scanned=meta.is_scanned,
            is_hvac_diagram=meta.is_hvac_diagram,
            extraction_confidence=round(meta.extraction_confidence, 3),
            total_blocks=len(doc.blocks),
            total_char_count=total_chars,
            quality_warnings=warnings,
            limitation_type=limitation_type,
            recommended_action=recommended_action,
        )
        self.write_record(record)
        return record

    def log_corrupted_file(
        self,
        relative_path: Union[str, Path],
        error_message: str,
        project_id: str = "UNKNOWN_PROJECT",
        building: str = "UNKNOWN_BUILDING",
        phase: str = "UNKNOWN_PHASE",
    ) -> LimitationRecord:
        """Log a corrupted or unreadable file that raised an exception during extraction."""
        rel = Path(relative_path)
        record = LimitationRecord(
            doc_id=f"corrupted_{rel.name}",
            file_name=rel.name,
            relative_folder_path=rel.parent.as_posix() if rel.parent != Path(".") else "",
            project_id=project_id,
            building=building,
            phase=phase,
            doc_category="unknown",
            extraction_method="failed",
            is_scanned=False,
            is_hvac_diagram=False,
            extraction_confidence=0.0,
            total_blocks=0,
            total_char_count=0,
            quality_warnings=[f"Extraction error: {error_message}"],
            limitation_type="corrupted_or_unreadable_file",
            recommended_action="Replace corrupted file from SharePoint archive.",
        )
        self.write_record(record)
        return record

    def write_record(self, record: LimitationRecord) -> None:
        """Append a single LimitationRecord as a JSON line to `limitations_report.jsonl`."""
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")

    def read_records(self) -> List[LimitationRecord]:
        """Read all logged limitation records from `limitations_report.jsonl`."""
        if not self.report_path.exists():
            return []
        records: List[LimitationRecord] = []
        for line in self.report_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data: Dict[str, Any] = json.loads(line)
                records.append(LimitationRecord.model_validate(data))
            except Exception:
                continue
        return records
