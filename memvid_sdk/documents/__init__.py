"""
Document Parsing Module

Auto-detects file type and parses PDF, XLSX, PPTX, DOCX documents.

Example:
    from memvid_sdk.documents import parse

    result = parse("./report.pdf")
    if result:
        print(f"{len(result['items'])} pages")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


class DocumentItem(TypedDict, total=False):
    """A single item (page/sheet/slide) from a document."""
    number: int
    name: Optional[str]
    title: Optional[str]
    text: str


class ParseResult(TypedDict):
    """Result of parsing a document."""
    type: str  # "pdf", "xlsx", "pptx", "docx"
    filename: str
    total_items: int
    items: List[DocumentItem]


class ParseOptions(TypedDict, total=False):
    """Options for document parsing."""
    max_items: Optional[int]


def parse(file_path: str, options: Optional[ParseOptions] = None) -> Optional[ParseResult]:
    """
    Parse a document file with automatic format detection.

    Supported formats:
    - PDF (.pdf) - per-page extraction
    - Excel (.xlsx, .xls) - per-sheet extraction
    - PowerPoint (.pptx, .ppt) - per-slide extraction
    - Word (.docx, .doc) - full document extraction

    Args:
        file_path: Path to the document file
        options: Parsing options

    Returns:
        ParseResult with items, or None for PDF if parser failed

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        from .pdf import parse_pdf
        return parse_pdf(file_path, options)
    elif ext in (".xlsx", ".xls"):
        from .xlsx import parse_xlsx
        return parse_xlsx(file_path, options)
    # For structured XLSX, use: from memvid_sdk.documents.xlsx import parse_xlsx_structured
    elif ext in (".pptx", ".ppt"):
        from .pptx import parse_pptx
        return parse_pptx(file_path, options)
    elif ext in (".docx", ".doc"):
        from .docx import parse_docx
        return parse_docx(file_path, options)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported: .pdf, .xlsx, .xls, .pptx, .ppt, .docx, .doc"
        )


def is_supported_format(file_path: str) -> bool:
    """Check if a file extension is supported for document parsing."""
    ext = Path(file_path).suffix.lower()
    return ext in (".pdf", ".xlsx", ".xls", ".pptx", ".ppt", ".docx", ".doc")


def get_document_type(file_path: str) -> Optional[str]:
    """Get the document type from a file path."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    elif ext in (".xlsx", ".xls"):
        return "xlsx"
    elif ext in (".pptx", ".ppt"):
        return "pptx"
    elif ext in (".docx", ".doc"):
        return "docx"
    return None


__all__ = [
    "parse",
    "is_supported_format",
    "get_document_type",
    "ParseResult",
    "ParseOptions",
    "DocumentItem",
    "parse_xlsx_structured",
]


def parse_xlsx_structured(file_path: str, **kwargs):
    """Convenience re-export of the structured XLSX extraction function."""
    from .xlsx import parse_xlsx_structured as _fn
    return _fn(file_path, **kwargs)
