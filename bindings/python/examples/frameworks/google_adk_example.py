#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile

from memvid_sdk import use
from memvid_sdk._sentinel import NoOp


def _tool_names(tools) -> list[str]:
    if isinstance(tools, NoOp):
        return []
    if isinstance(tools, list):
        return [str(item.get("name")) for item in tools if isinstance(item, dict) and item.get("name")]
    # google.genai.types.Tool (optional dependency) – keep output stable without importing
    return []


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="memvid-google-adk-") as tmpdir:
        path = os.path.join(tmpdir, "google_adk.mv2")
        with use("google-adk", path, mode="create", enable_lex=True, enable_vec=False) as mv:
            mv.put("Hello", "note", {}, text="Hello Memvid", enable_embedding=False)
            res = mv.find("Hello", mode="lex", k=3)
            print(
                {
                    "kind": "google-adk",
                    "tools": _tool_names(mv.tools),
                    "tools_type": type(mv.tools).__name__,
                    "hits": res["total_hits"],
                }
            )


if __name__ == "__main__":
    main()

