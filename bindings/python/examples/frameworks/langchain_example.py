#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile

from memvid_sdk import use
from memvid_sdk._sentinel import NoOp


def _tool_count(tools) -> int:
    if isinstance(tools, NoOp):
        return 0
    if isinstance(tools, list):
        return len(tools)
    return 0


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="memvid-langchain-") as tmpdir:
        path = os.path.join(tmpdir, "langchain.mv2")
        with use("langchain", path, mode="create", enable_lex=True, enable_vec=False) as mv:
            mv.put("Hello", "note", {}, text="Hello Memvid", enable_embedding=False)
            res = mv.find("Hello", mode="lex", k=3)
            print(
                {
                    "kind": "langchain",
                    "available": not isinstance(mv.tools, NoOp),
                    "tools": _tool_count(mv.tools),
                    "hits": res["total_hits"],
                }
            )


if __name__ == "__main__":
    main()

