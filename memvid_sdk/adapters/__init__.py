"""Adapter loader registrations for the Memvid Python SDK."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .._registry import registry
from .._sentinel import NoOp


def _unavailable(kind: str, reason: str) -> Mapping[str, Any]:
    key = kind.replace("_", "-")
    return {
        "tools": NoOp(
            f"{kind} adapters unavailable: {reason}", f"memvid.adapters.{key}.tools"
        ),
        "functions": NoOp(
            f"{kind} adapters unavailable: {reason}", f"memvid.adapters.{key}.functions"
        ),
        "nodes": NoOp(
            f"{kind} adapters unavailable: {reason}", f"memvid.adapters.{key}.nodes"
        ),
        "as_query_engine": None,
    }


def _basic_loader(_core: Optional[object], _apikey: Optional[Mapping[str, str]]):
    return {
        "tools": NoOp("basic kind exposes no tools", "memvid.adapters.basic.tools"),
        "functions": [],
        "nodes": NoOp("basic kind exposes no nodes", "memvid.adapters.basic.nodes"),
        "as_query_engine": None,
    }


registry.register("basic", _basic_loader)

# Import side-effect registrations for optional adapters. The modules themselves
# guard their imports so simply importing here won't trigger heavy dependencies.
from . import (  # noqa: F401  (imported for side effects only)
    autogen,
    crewai,
    google_adk,
    haystack,
    langchain,
    langgraph,
    llamaindex,
    mcp,
    openai,
    semantic_kernel,
    vercel_ai,
)

__all__ = ["_unavailable"]

