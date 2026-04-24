"""Shared pytest fixtures for the ``memvid_sdk`` test suite.

Fixtures defined here are discovered automatically by pytest for every
test file under ``tests/`` and do not need to be imported.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator

import pytest

import memvid_sdk


@pytest.fixture(autouse=True)
def _isolate_global_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Guarantee each test sees a clean SDK config and no Memvid env vars.

    Snapshots whatever global config was in place, clears it for the
    test, and restores it on teardown so ordering between tests (and
    between this module and the rest of the suite) cannot leak state.
    """
    monkeypatch.delenv("MEMVID_API_KEY", raising=False)
    monkeypatch.delenv("MEMVID_DASHBOARD_URL", raising=False)
    monkeypatch.delenv("MEMVID_API_URL", raising=False)

    previous_config: Dict[str, Any] = dict(memvid_sdk.get_config() or {})
    memvid_sdk.reset_config()
    try:
        yield
    finally:
        memvid_sdk.reset_config()
        if previous_config:
            memvid_sdk.configure(previous_config)  # type: ignore[arg-type]
