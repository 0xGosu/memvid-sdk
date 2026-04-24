"""LangGraph adapter placeholder."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .._registry import registry
from . import _unavailable


def _load_langgraph(
    _core: Optional[Any],
    _apikey: Optional[Mapping[str, str]],
):
    return _unavailable("langgraph", "install 'langgraph' to enable this integration")


registry.register("langgraph", _load_langgraph)

