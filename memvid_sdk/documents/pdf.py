"""PDF Parser using pypdf."""

from __future__ import annotations

import os
from typing import Optional

from . import ParseOptions, ParseResult, DocumentItem


def parse_pdf(file_path: str, options: Optional[ParseOptions] = None) -> Optional[ParseResult]:
    """
    Parse a PDF file, extracting text per page.

    Args:
        file_path: Path to the PDF file
        options: Parsing options (max_items limits pages)

    Returns:
        ParseResult with per-page items, or None if parsing fails
    """
    filename = os.path.basename(file_path)

    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            print(f"[memvid] pypdf/PyPDF2 not installed, PDF parsing unavailable")
            return None

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        max_items = (options or {}).get("max_items") or total_pages

        items: list[DocumentItem] = []
        for i in range(min(total_pages, max_items)):
            page = reader.pages[i]
            text = page.extract_text() or ""
            if text.strip():
                items.append({
                    "number": i + 1,
                    "text": text,
                })

        return {
            "type": "pdf",
            "filename": filename,
            "total_items": total_pages,
            "items": items,
        }
    except Exception as e:
        print(f"[memvid] PDF parsing failed for {filename}: {e}")
        return None
