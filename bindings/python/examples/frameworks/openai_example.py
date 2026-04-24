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
        out: list[str] = []
        for item in tools:
            if isinstance(item, dict):
                out.append(str((item.get("function") or {}).get("name") or ""))
        return [name for name in out if name]
    return []


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="memvid-openai-") as tmpdir:
        path = os.path.join(tmpdir, "openai.mv2")
        with use("openai", path, mode="create", enable_lex=True, enable_vec=False) as mv:
            mv.put("Hello", "note", {}, text="Hello Memvid", enable_embedding=False)
            res = mv.find("Hello", mode="lex", k=3)
            print({"kind": "openai", "tools": _tool_names(mv.tools), "hits": res["total_hits"]})


if __name__ == "__main__":
    main()

