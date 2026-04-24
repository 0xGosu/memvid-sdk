"""Adapter registry and lazy loader plumbing for the Memvid Python SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

from ._sentinel import NoOp


Loader = Callable[[Optional[Any], Optional[Mapping[str, str]]], Mapping[str, Any]]


@dataclass(frozen=True)
class RegistryEntry:
    """Represents a registered adapter loader for a given kind."""

    kind: str
    loader: Loader


class AdapterRegistry:
    """Lazy registry mapping kinds to adapter loader callables."""

    def __init__(self) -> None:
        self._loaders: Dict[str, Loader] = {}
        self._failures: MutableMapping[str, Mapping[str, Any]] = {}
        self._cache: MutableMapping[str, Mapping[str, Any]] = {}

    def register(self, kind: str, loader: Loader) -> None:
        key = kind.lower()
        if key in self._loaders:
            raise ValueError(f"adapter kind '{kind}' already registered")
        self._loaders[key] = loader

    def resolve(
        self, kind: str, core: Optional[Any], apikey: Optional[Mapping[str, str]]
    ) -> Mapping[str, Any]:
        key = kind.lower()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if key in self._failures:
            return self._failures[key]

        loader = self._loaders.get(key)
        if loader is None:
            result: Mapping[str, Any] = {
                "tools": NoOp(f"kind '{kind}' is not supported", "memvid.adapters.unsupported"),
                "functions": NoOp(
                    f"kind '{kind}' is not supported", "memvid.adapters.unsupported"
                ),
                "nodes": NoOp(f"kind '{kind}' is not supported", "memvid.adapters.unsupported"),
                "as_query_engine": None,
            }
            self._failures[key] = result
            return result

        try:
            loaded = loader(core, apikey)
        except Exception as exc:  # pragma: no cover - defensive
            loaded = {
                "tools": NoOp(
                    f"failed to load adapter '{kind}': {exc}",
                    f"memvid.adapters.{key}.tools",
                ),
                "functions": NoOp(
                    f"failed to load adapter '{kind}': {exc}",
                    f"memvid.adapters.{key}.functions",
                ),
                "nodes": NoOp(
                    f"failed to load adapter '{kind}': {exc}",
                    f"memvid.adapters.{key}.nodes",
                ),
                "as_query_engine": None,
            }
        else:
            self._cache[key] = loaded
        return loaded


registry = AdapterRegistry()
