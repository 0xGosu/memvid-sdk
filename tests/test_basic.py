"""Basic smoke tests for the ``memvid_sdk`` Python package.

These tests exercise the functionality that works without a
``MEMVID_API_KEY`` set. They confirm:

* the Rust extension loads,
* the advertised public surface is present,
* a local ``.mv2`` file can be created, inspected, closed and reopened
  read-only,
* a fresh memory is provisioned at exactly the free-tier capacity
  ceiling,
* a ``put()`` within the free-tier ceiling succeeds without an API key,
* a ``put()`` followed by ``find()`` returns the indexed frame as a hit,
* opening a missing file raises ``memvid_sdk.FileNotFoundError``.

Run with ``pytest`` from the project root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator, Set

import pytest

import memvid_sdk


# The SDK's free-tier capacity ceiling. Mirrors ``FREE_TIER_LIMIT_BYTES``
# in ``bindings/python/src/lib.rs:70`` (``9999 * 1024 ** 3``). Duplicated
# here on purpose: the Rust constant is private to the binding and the
# test suite needs to assert against it directly.
FREE_TIER_CEILING_BYTES: int = 9999 * (1 << 30)


# The ``_isolate_global_state`` autouse fixture that snapshots and
# restores the SDK's global config + env vars around each test lives in
# ``tests/conftest.py`` so the basic and advanced suites share a single
# source of truth.


# ---------------------------------------------------------------------------
# Module-level / import sanity
# ---------------------------------------------------------------------------


def test_native_extension_loads() -> None:
    """The compiled ``_lib`` extension is importable and is a native module."""
    from memvid_sdk import _lib  # imported lazily to surface loader errors here

    assert _lib.__file__ is not None
    assert _lib.__file__.endswith((".so", ".pyd", ".dylib")), (
        f"_lib is not a compiled extension: {_lib.__file__}"
    )


def test_public_api_surface_is_present() -> None:
    """Key public symbols documented in ``__all__`` exist on the package."""
    expected: Set[str] = {
        "Memvid",
        "use",
        "create",
        "info",
        "MemvidError",
        "ApiKeyRequiredError",
        "CapacityExceededError",
        "FileNotFoundError",
        "CorruptFileError",
        "LexIndexDisabledError",
        "LockedError",
    }
    missing = {name for name in expected if not hasattr(memvid_sdk, name)}
    assert not missing, f"public symbols missing from memvid_sdk: {sorted(missing)}"


def test_error_classes_derive_from_memvid_error() -> None:
    """All domain-specific errors inherit from ``MemvidError``."""
    for cls_name in (
        "ApiKeyRequiredError",
        "CapacityExceededError",
        "CorruptFileError",
        "FileNotFoundError",
        "LexIndexDisabledError",
        "LockedError",
    ):
        cls = getattr(memvid_sdk, cls_name)
        assert issubclass(cls, memvid_sdk.MemvidError), (
            f"{cls_name} does not derive from MemvidError"
        )


def test_info_returns_diagnostic_dict() -> None:
    """``info()`` returns diagnostics with the documented keys."""
    info = memvid_sdk.info()
    assert isinstance(info, dict)
    for key in ("sdk_version", "platform", "python", "native_exports"):
        assert key in info, f"info() missing key: {key}"
    assert isinstance(info["native_exports"], list)
    assert len(info["native_exports"]) > 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mv2_path(tmp_path: Path) -> Path:
    """Return a fresh ``.mv2`` path in a pytest-managed temp directory."""
    return tmp_path / "test.mv2"


@pytest.fixture
def memvid(mv2_path: Path) -> Iterator[memvid_sdk.Memvid]:
    """Yield a freshly created ``Memvid`` handle and close it on teardown."""
    mv = memvid_sdk.create(str(mv2_path))
    try:
        yield mv
    finally:
        mv.close()


# ---------------------------------------------------------------------------
# Local memory lifecycle (no API key required)
# ---------------------------------------------------------------------------


def test_create_writes_a_non_empty_file(mv2_path: Path, memvid: memvid_sdk.Memvid) -> None:
    """``create()`` produces a non-empty ``.mv2`` file on disk."""
    assert mv2_path.exists()
    assert mv2_path.stat().st_size > 0


def test_create_returns_memvid_handle(memvid: memvid_sdk.Memvid) -> None:
    """``create()`` returns a ``memvid_sdk.Memvid`` instance."""
    assert isinstance(memvid, memvid_sdk.Memvid)


def test_fresh_memory_capacity_matches_free_tier_ceiling(memvid: memvid_sdk.Memvid) -> None:
    """A freshly created memory is provisioned at the free-tier ceiling.

    Both ``stats()["capacity_bytes"]`` and ``get_capacity()`` must
    return ``FREE_TIER_CEILING_BYTES``. Asserting the exact value (not
    just ``> 1 GiB``) makes this test coherent with
    ``test_put_within_free_tier_succeeds_without_api_key``, which
    depends on capacity being at or below the ceiling for the keyless
    write path.
    """
    stats_capacity = memvid.stats()["capacity_bytes"]
    reported_capacity = memvid.get_capacity()

    assert isinstance(stats_capacity, int)
    assert isinstance(reported_capacity, int)

    assert stats_capacity == FREE_TIER_CEILING_BYTES, (
        f"expected stats()['capacity_bytes'] == free-tier ceiling "
        f"({FREE_TIER_CEILING_BYTES}), got {stats_capacity}"
    )
    assert reported_capacity == FREE_TIER_CEILING_BYTES, (
        f"expected get_capacity() == free-tier ceiling "
        f"({FREE_TIER_CEILING_BYTES}), got {reported_capacity}"
    )
    assert reported_capacity == stats_capacity, (
        f"get_capacity() ({reported_capacity}) diverged from "
        f"stats()['capacity_bytes'] ({stats_capacity})"
    )


def test_stats_reports_expected_keys_on_fresh_memory(memvid: memvid_sdk.Memvid) -> None:
    """``stats()`` on a fresh memory reports frame counts and index flags."""
    stats: Dict[str, Any] = memvid.stats()

    assert isinstance(stats, dict)
    expected_keys: Set[str] = {
        "frame_count",
        "active_frame_count",
        "capacity_bytes",
        "has_lex_index",
        "has_vec_index",
        "has_time_index",
    }
    missing = expected_keys - set(stats.keys())
    assert not missing, f"stats() missing keys: {sorted(missing)}"

    assert stats["frame_count"] == 0
    assert stats["active_frame_count"] == 0


def test_enable_lex_flips_has_lex_index(memvid: memvid_sdk.Memvid) -> None:
    """``enable_lex()`` flips the ``has_lex_index`` flag on stats."""
    memvid.enable_lex()
    stats = memvid.stats()
    assert stats["has_lex_index"] is True


def test_enable_vec_flips_has_vec_index(memvid: memvid_sdk.Memvid) -> None:
    """``enable_vec()`` flips the ``has_vec_index`` flag on stats."""
    memvid.enable_vec()
    stats = memvid.stats()
    assert stats["has_vec_index"] is True


def test_find_on_empty_memory_returns_no_hits(memvid: memvid_sdk.Memvid) -> None:
    """``find()`` on an empty memory returns a result dict with an empty ``hits`` list."""
    result = memvid.find("anything")

    assert isinstance(result, dict)
    assert "hits" in result, f"find() result missing 'hits' key: {sorted(result.keys())}"
    assert result["hits"] == []


def test_put_then_find_returns_hit_for_indexed_text(memvid: memvid_sdk.Memvid) -> None:
    """A ``put()`` followed by ``find()`` returns the indexed frame as a hit.

    End-to-end memory-hit path on the free tier:

    * ``enable_lex()`` turns on the lexical index so text is tokenized,
    * ``put()`` ingests a document carrying a deliberately unique term,
    * ``seal()`` flushes pending index writes so the term is queryable,
    * ``find()`` with that unique term returns exactly one hit whose
      ``frame_id`` matches the id returned by ``put()`` and whose
      ``title`` matches the one that was ingested.

    The query term (``zynquilateralxyzzy``) is intentionally nonsensical
    to rule out collisions with any incidental tokenizer vocabulary or
    stopword list.
    """
    memvid.enable_lex()

    unique_term = "zynquilateralxyzzy"
    frame_id = memvid.put(
        title="hit-target",
        label="basic",
        text=f"this document mentions {unique_term} exactly once",
    )
    memvid.seal()

    result = memvid.find(unique_term)

    assert isinstance(result, dict)
    assert "hits" in result, (
        f"find() result missing 'hits' key: {sorted(result.keys())}"
    )
    hits = result["hits"]
    assert len(hits) == 1, (
        f"expected exactly one hit for unique term {unique_term!r}, got {hits}"
    )

    hit = hits[0]
    # ``put()`` returns a decimal-string id (see
    # ``test_put_within_free_tier_succeeds_without_api_key``) while
    # ``find()`` hits expose ``frame_id`` as an int — normalize both
    # sides to int for the identity check, matching the style used in
    # ``test_advance.py``.
    assert int(hit["frame_id"]) == int(frame_id), (
        f"hit frame_id {hit['frame_id']!r} != put frame_id {frame_id!r}"
    )
    assert hit["title"] == "hit-target"


def test_memories_and_list_tables_are_empty(memvid: memvid_sdk.Memvid) -> None:
    """A fresh memory has no memory cards and no tables."""
    assert memvid.memories() == {"cards": [], "count": 0}
    assert memvid.list_tables() == []


def test_reopen_read_only(mv2_path: Path) -> None:
    """A file created with ``create()`` can be reopened read-only."""
    created = memvid_sdk.create(str(mv2_path))
    created.close()

    mv = memvid_sdk.use("basic", str(mv2_path), mode="open", read_only=True)
    try:
        assert isinstance(mv, memvid_sdk.Memvid)
        stats = mv.stats()
        assert stats["frame_count"] == 0
    finally:
        mv.close()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_open_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """Opening a non-existent file raises ``memvid_sdk.FileNotFoundError``."""
    missing = tmp_path / "definitely-not-here.mv2"
    with pytest.raises(memvid_sdk.FileNotFoundError):
        memvid_sdk.use("basic", str(missing), mode="open")


def test_put_within_free_tier_succeeds_without_api_key(memvid: memvid_sdk.Memvid) -> None:
    """A ``put()`` on a fresh file succeeds without an API key.

    ``create()`` provisions a fresh memory at exactly the free-tier
    ceiling (see ``FREE_TIER_CEILING_BYTES``), so a write without an
    API key is allowed by design. This test confirms:

    * no ``MEMVID_API_KEY`` is configured (surfaces the no-key
      precondition at the call site),
    * capacity sits at or below the free-tier ceiling,
    * ``put()`` returns a decimal-string frame id equal to the
      pre-``put`` frame count (the documented TOC-index contract), and
    * ``frame_count`` / ``active_frame_count`` advance by one.
    """
    assert os.environ.get("MEMVID_API_KEY") is None, (
        "autouse fixture must have cleared MEMVID_API_KEY"
    )

    capacity = memvid.stats()["capacity_bytes"]
    assert capacity <= FREE_TIER_CEILING_BYTES, (
        f"precondition failed: expected capacity_bytes ({capacity}) <= "
        f"free-tier ceiling ({FREE_TIER_CEILING_BYTES}); gate would block a keyless write"
    )

    pre_stats = memvid.stats()
    frame_id = memvid.put(title="doc-1", label="basic", text="hello world")

    assert isinstance(frame_id, str) and frame_id.isdigit(), (
        f"put() should return a decimal-string frame id, got {frame_id!r}"
    )
    assert int(frame_id) == pre_stats["frame_count"], (
        f"frame_id {frame_id!r} should equal pre-put frame_count "
        f"({pre_stats['frame_count']})"
    )

    post_stats = memvid.stats()
    assert post_stats["frame_count"] == pre_stats["frame_count"] + 1
    assert post_stats["active_frame_count"] == pre_stats["active_frame_count"] + 1
