#!/usr/bin/env python3
"""
Simple test script for capacity enforcement in Python SDK.

Usage:
    python3 test_capacity_simple.py
    python3 test_capacity_simple.py --api-key memvidapi_xxxxx
"""

import argparse
import os
import sys
from memvid_sdk import use

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test capacity enforcement in Python SDK")
    parser.add_argument("--api-key", type=str, help="API key to test with")
    args = parser.parse_args()

    print("=" * 60)
    print("🧪 Simple Capacity Enforcement Test")
    print("=" * 60)

    # Use existing file
    test_file = "huge_file.mv2"

    if not os.path.exists(test_file):
        print(f"❌ File not found: {test_file}")
        return

    # Check file size
    file_size = os.path.getsize(test_file)
    file_size_mb = file_size / (1024**2)
    file_size_gb = file_size / (1024**3)
    print(f"\n📁 File: {test_file}")
    print(f"   Size: {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("MEMVID_API_KEY")

    # Backup and clear environment variable
    api_key_backup = os.environ.get("MEMVID_API_KEY")
    if "MEMVID_API_KEY" in os.environ:
        del os.environ["MEMVID_API_KEY"]

    print("\n" + "=" * 60)
    print("Test 1: Read file without API key")
    print("=" * 60)
    try:
        mem = use('basic', test_file, read_only=True)
        result = mem.find("test")
        print(f"✅ Read successful - found {len(result.get('hits', []))} results")
        del mem  # Close the file handle
    except Exception as e:
        print(f"❌ Read failed: {e}")

    print("\n" + "=" * 60)
    print("Test 2: Write to file without API key")
    print("=" * 60)
    if file_size_gb > 1.0:
        print(f"   File is {file_size_gb:.2f} GB (> 1GB free tier)")
        print(f"   Expected: Write should FAIL")
    else:
        print(f"   File is {file_size_mb:.2f} MB (< 1GB free tier)")
        print(f"   Expected: Write should SUCCEED")

    mem = None
    try:
        mem = use('basic', test_file)
        mem.put("Capacity Test Entry", "test", {}, text="Testing capacity enforcement without API key")

        if file_size_gb > 1.0:
            print(f"❌ Write succeeded (should have failed for >1GB file!)")
        else:
            print(f"✅ Write succeeded (file < 1GB, no API key needed)")
    except Exception as e:
        error_msg = str(e)
        if file_size_gb > 1.0 and ("exceeds 1GB free tier limit" in error_msg or "MEMVID_API_KEY" in error_msg):
            print(f"✅ Correctly blocked write: {error_msg}")
        elif file_size_gb <= 1.0:
            print(f"❌ Write failed unexpectedly: {error_msg}")
        else:
            print(f"⚠️  Error: {error_msg}")
    finally:
        if mem is not None:
            del mem  # Ensure file handle is closed

    # Test with API key if available
    if api_key:
        print(f"\n📌 Using API key: {api_key[:15]}...")

        print("\n" + "=" * 60)
        print("Test 3: Write with API key from environment")
        print("=" * 60)
        os.environ["MEMVID_API_KEY"] = api_key

        mem = None
        try:
            mem = use('basic', test_file)
            mem.put("Test with API Key", "test", {}, text="Testing with API key")
            print(f"✅ Write with API key succeeded (within plan capacity)")
        except Exception as e:
            error_msg = str(e)
            if "exceeds plan limit" in error_msg:
                print(f"✅ Correctly blocked (exceeds plan limit): {error_msg}")
            else:
                print(f"❌ Write failed: {error_msg}")
        finally:
            if mem is not None:
                del mem

        # Test API key as parameter
        print("\n" + "=" * 60)
        print("Test 4: Write with API key as parameter")
        print("=" * 60)
        del os.environ["MEMVID_API_KEY"]

        mem = None
        try:
            mem = use('basic', test_file, apikey=api_key)
            mem.put("Test API key param", "test", {}, text="Testing API key parameter")
            print(f"✅ API key parameter works (within plan capacity)")
        except Exception as e:
            error_msg = str(e)
            if "exceeds plan limit" in error_msg:
                print(f"✅ Correctly blocked (exceeds plan limit): {error_msg}")
            else:
                print(f"❌ Failed: {error_msg}")
        finally:
            if mem is not None:
                del mem
    else:
        print("\n⏭️  Skipping API key tests (no API key provided)")
        print("   Usage: python3 test_capacity_simple.py --api-key memvidapi_xxxxx")

    print("\n" + "=" * 60)
    print("✨ Tests completed!")
    print("=" * 60)

    # Restore env var
    if api_key_backup:
        os.environ["MEMVID_API_KEY"] = api_key_backup

if __name__ == "__main__":
    main()
