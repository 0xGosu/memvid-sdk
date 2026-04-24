"""Semantic Kernel adapter placeholder."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .._registry import registry
from . import _unavailable


def _load_semantic_kernel(
    _core: Optional[Any],
    _apikey: Optional[Mapping[str, str]],
):
    return _unavailable(
        "semantic_kernel",
        "install 'semantic-kernel' to enable this integration",
    )


registry.register("semantic-kernel", _load_semantic_kernel)

