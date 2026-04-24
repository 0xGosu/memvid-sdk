"""Helper utilities for adapter fallbacks."""

from __future__ import annotations

import threading
import warnings
from typing import Any, Iterable


_warned_keys: set[str] = set()
_lock = threading.Lock()


class NoOp:
    """Sentinel used when optional adapters are unavailable.

    The sentinel warns the first time it is interacted with, evaluates to ``False``
    in boolean contexts and remains callable without side effects.
    """

    def __init__(self, message: str, key: str) -> None:
        self._message = message
        self._key = key

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        self._warn_once()
        return False

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self._warn_once()
        return None

    def __iter__(self) -> Iterable[None]:  # pragma: no cover - defensive
        self._warn_once()
        return iter(())

    def __getattr__(self, name: str) -> "NoOp":
        self._warn_once()
        return self

    def _warn_once(self) -> None:
        with _lock:
            if self._key in _warned_keys:
                return
            warnings.warn(self._message, RuntimeWarning, stacklevel=3)
            _warned_keys.add(self._key)


__all__ = ["NoOp"]

