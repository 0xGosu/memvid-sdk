"""Vercel AI SDK adapter placeholder."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .._registry import registry
from . import _unavailable


def _load_vercel_ai(
    _core: Optional[Any],
    _apikey: Optional[Mapping[str, str]],
):
    return _unavailable(
        "vercel_ai",
        "install 'ai' (Vercel AI SDK) to enable this integration",
    )


registry.register("vercel-ai", _load_vercel_ai)

