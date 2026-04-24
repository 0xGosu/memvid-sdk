#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile

from memvid_sdk import use
from memvid_sdk._sentinel import NoOp


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="memvid-autogen-") as tmpdir:
        path = os.path.join(tmpdir, "autogen.mv2")
        with use("autogen", path, mode="create", enable_lex=True, enable_vec=False) as mv:
            mv.put("Hello", "note", {}, text="Hello Memvid", enable_embedding=False)
            res = mv.find("Hello", mode="lex", k=3)
            print(
                {
                    "kind": "autogen",
                    "available": not isinstance(mv.tools, NoOp),
                    "hits": res["total_hits"],
                }
            )


if __name__ == "__main__":
    main()

