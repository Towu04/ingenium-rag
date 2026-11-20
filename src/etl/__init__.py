"""Stage 1: SharePoint/local extraction and 3-layer metadata resolution emitting `domain.ParsedDoc`."""

from etl.catalog import DocumentRegisterIndex, ProjectCatalog, is_document_register_file
from etl.extractors import DocxExtractor, PdfExtractor, XlsxExtractor, inspect_pdf_document
from etl.metadata import MatchedMetadata, MetadataMatcher
from etl.quality import LimitationRecord, QualityLogger, score_callout_readability
from etl.runner import ETLRunner, ETLRunResult, ManifestManager, load_processed_docs
from etl.sharepoint import (
    RemoteDriveItem,
    SharePointConfig,
    SharePointConnector,
    compute_file_sha256,
)

__all__ = [
    "DocumentRegisterIndex",
    "DocxExtractor",
    "ETLRunResult",
    "ETLRunner",
    "LimitationRecord",
    "ManifestManager",
    "MatchedMetadata",
    "MetadataMatcher",
    "PdfExtractor",
    "ProjectCatalog",
    "QualityLogger",
    "RemoteDriveItem",
    "SharePointConfig",
    "SharePointConnector",
    "XlsxExtractor",
    "compute_file_sha256",
    "inspect_pdf_document",
    "is_document_register_file",
    "load_processed_docs",
    "score_callout_readability",
]
