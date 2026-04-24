"""Haystack adapter placeholder."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .._registry import registry
from . import _unavailable


def _load_haystack(
    _core: Optional[Any],
    _apikey: Optional[Mapping[str, str]],
):
    return _unavailable("haystack", "install 'haystack-ai' to enable this integration")


registry.register("haystack", _load_haystack)

