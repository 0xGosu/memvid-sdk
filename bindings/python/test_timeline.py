#!/usr/bin/env python3
"""Test put, put_many, and timeline functionality."""

import os
from memvid_sdk import create, use

TEST_FILE = "test_python.mv2"

def main():
    # Clean up
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)

    print("1. Creating memory file...")
    mv = create(TEST_FILE)

    print("2. Enabling lex...")
    mv.enable_lex()

    print("3. Storing first doc with put()...")
    mv.put(title="First Doc", label="first", metadata={}, text="First document")

    print("3a. Seal after put()...")
    mv.seal()

    stats = mv.stats()
    print(f"   has_time_index: {stats.get('has_time_index', False)}")

    print("4. Testing put_many()...")
    frame_ids = mv.put_many([
        {"title": "Second Doc", "label": "second", "text": "Second document"},
        {"title": "Third Doc", "label": "third", "text": "Third document"},
    ])
    print(f"   Batch stored as frames: {frame_ids}")

    print("4a. Stats after put_many:")
    stats = mv.stats()
    print(f"   has_time_index: {stats.get('has_time_index', False)}")

    print("5. Final seal...")
    mv.seal()

    print("5a. Stats after final seal:")
    stats = mv.stats()
    print(f"   has_time_index: {stats.get('has_time_index', False)}")

    print("\n6. Testing timeline()...")
    timeline = mv.timeline(limit=10)
    print(f"   Got {len(timeline)} entries:")
    for entry in timeline:
        print(f"   - Frame {entry['frame_id']}: {entry.get('preview', 'N/A')[:50]}")

    print("\nDone!")

if __name__ == "__main__":
    main()
