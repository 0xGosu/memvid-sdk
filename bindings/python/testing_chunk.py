#!/usr/bin/env python3
"""Test PDF chunking with the Python SDK and Gemini API."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from pprint import pprint

# Add the SDK to path
sys.path.insert(0, os.path.dirname(__file__))

from memvid_sdk import Memvid, create, use

# Test file paths
MV2_PATH = Path("/tmp/test-pdf-chunks.mv2")
PDF_PATH = Path.home() / "Desktop" / "sp-global-impact-report-2024.pdf"

# Gemini API key
os.environ["GEMINI_API_KEY"] = "AIzaSyAPnKEN72SwGAQfb4JiUhvk2g60qFg9zfQ"


def main() -> None:
    print("=" * 60)
    print("Testing PDF Chunking with Python SDK")
    print("=" * 60)

    # Check PDF exists
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        return

    print(f"\n1. Creating memory at {MV2_PATH}")

    # Remove old file if exists
    if MV2_PATH.exists():
        MV2_PATH.unlink()

    # Create memory
    mv: Memvid = create(str(MV2_PATH))
    mv.enable_lex()
    print(f"   Created: {MV2_PATH}")

    print(f"\n2. Ingesting PDF: {PDF_PATH}")
    mv.put(
        title="SP Global Impact Report 2024",
        label="report",
        metadata={"year": "2024", "company": "S&P Global"},
        file=str(PDF_PATH),
    )
    print("   PDF ingested successfully")

    # Get stats
    stats = mv.stats()
    print(f"\n3. Memory stats:")
    pprint(stats)

    print("\n4. Testing search for 'external hires'...")
    results = mv.find("external hires", k=3)
    print(f"   Found {len(results)} results:")
    # Results might be a dict with 'hits' key or a list
    hits = results.get("hits", results) if isinstance(results, dict) else list(results)
    for i, hit in enumerate(hits[:3], 1):
        if isinstance(hit, dict):
            title = hit.get("title", "N/A")
            score = hit.get("score", 0)
            text_preview = hit.get("text", "")[:150].replace("\n", " ") + "..."
            print(f"   [{i}] {title} (score: {score:.3f})")
            print(f"       {text_preview}")
        else:
            print(f"   [{i}] {hit}")

    print("\n5. Testing ASK with Gemini API...")
    print("   Question: How many external hires did S&P Global make in 2024?")

    try:
        answer = mv.ask(
            "How many external hires did S&P Global make in 2024?",
            model="gemini-2.0-flash",
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        print(f"\n   Answer:")
        if isinstance(answer, dict):
            pprint(answer)
        else:
            print(f"   {answer}")
    except Exception as e:
        print(f"   ERROR with ask: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
