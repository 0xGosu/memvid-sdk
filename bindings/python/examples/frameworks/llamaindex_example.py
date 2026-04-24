#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile

from memvid_sdk import use
from memvid_sdk._sentinel import NoOp


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="memvid-llamaindex-") as tmpdir:
        path = os.path.join(tmpdir, "llamaindex.mv2")
        with use("llamaindex", path, mode="create", enable_lex=True, enable_vec=False) as mv:
            mv.put("Hello", "note", {}, text="Hello Memvid", enable_embedding=False)
            res = mv.find("Hello", mode="lex", k=3)
            has_qe = mv.as_query_engine is not None
            print(
                {
                    "kind": "llamaindex",
                    "available": not isinstance(mv.tools, NoOp),
                    "has_query_engine": bool(has_qe),
                    "hits": res["total_hits"],
                }
            )


if __name__ == "__main__":
    main()

