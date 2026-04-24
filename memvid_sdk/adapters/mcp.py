"""Memvid MCP client adapter placeholder."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .._registry import registry
from . import _unavailable


def _load_mcp(
    _core: Optional[Any],
    _apikey: Optional[Mapping[str, str]],
):
    return _unavailable("mcp", "MCP integrations are client-specific; provide your own client")


registry.register("mcp", _load_mcp)

