"""Document extractors emitting canonical `domain.ParsedDoc` directly."""

from etl.extractors.docx import DocxExtractor
from etl.extractors.pdf import PdfExtractor, inspect_pdf_document
from etl.extractors.xlsx import XlsxExtractor

__all__ = ["DocxExtractor", "PdfExtractor", "XlsxExtractor", "inspect_pdf_document"]
