"""Comprehensive functional tests for the ``memvid_sdk`` Python package.

Where ``test_basic.py`` is a smoke suite, this file exercises the main
end-user flows in more depth:

* **Ingestion**: ``put()``, ``put_many()`` with metadata, tags, labels.
* **Search**: ``find()`` result shape, ranking, ``k``-limit, no-match.
* **Timeline**: ordering, reverse, ``limit``.
* **Memory cards**: ``add_memory_cards()`` + ``memories()`` + stats.
* **Sessions**: ``session_start`` / ``session_checkpoint`` /
  ``session_end`` / ``session_list`` lifecycle.
* **Removal**: ``remove()`` purges a frame from search results.
* **Durability**: ``seal()``, ``commit()``, close-reopen round-trip,
  use-after-close guard.
* **Integrity**: ``verify()``, ``doctor()``, ``export_facts()``.
* **Module-level**: ``configure`` / ``get_config`` / ``reset_config``
  round-trip, ``validate_config``, ``lock_who`` / ``lock_nudge`` on an
  unlocked file.

All tests run offline without ``MEMVID_API_KEY`` — ingestion sits below
the free-tier capacity ceiling by design.
"""

from __future__ import annotations

import json
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set

import pytest

import memvid_sdk


# The ``_isolate_global_state`` autouse fixture that clears SDK config
# and Memvid env vars around every test is defined in
# ``tests/conftest.py`` and is shared with ``test_basic.py``.


# Target on-disk file size for the free-tier stress test: 60 MB,
# power-of-two form. Past the old 50 MB free-tier ceiling but trivial
# against the current 9999 GB ceiling — the stress test fills to this
# size to confirm the lifted ceiling actually lets writes through.
STRESS_TARGET_FILE_SIZE_BYTES: int = 60 * (1 << 20)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mv2_path(tmp_path: Path) -> Path:
    """Path to a fresh ``.mv2`` file in a pytest-managed temp directory."""
    return tmp_path / "advance.mv2"


@pytest.fixture
def memvid(mv2_path: Path) -> Iterator[memvid_sdk.Memvid]:
    """Create a fresh memory with lex enabled and close it on teardown."""
    mv = memvid_sdk.create(str(mv2_path), enable_lex=True, enable_vec=False)
    try:
        yield mv
    finally:
        if not mv.closed:
            mv.close()


@pytest.fixture
def seeded_memvid(memvid: memvid_sdk.Memvid) -> memvid_sdk.Memvid:
    """A memory pre-populated with three distinctly-worded documents.

    Frame 0: ``alpha gamma`` (tag: ``tag1``)
    Frame 1: ``beta delta``
    Frame 2: ``gamma gamma gamma gamma`` — wide term-frequency gap over
    frame 0 so top-hit assertions stay stable across minor scoring
    changes (length normalization, idf shifts, etc.).
    """
    memvid.put(
        title="Alpha doc",
        label="docs",
        text="alpha gamma",
        metadata={"src": "unit-test", "version": 1},
        tags=["tag1", "tag2"],
    )
    memvid.put(title="Beta doc", label="docs", text="beta delta")
    memvid.put(title="Gamma doc", label="news", text="gamma gamma gamma gamma")
    memvid.seal()
    return memvid


# ---------------------------------------------------------------------------
# Ingestion: put, put_many
# ---------------------------------------------------------------------------


def test_put_returns_sequential_frame_ids(memvid: memvid_sdk.Memvid) -> None:
    """Successive ``put()`` calls return decimal-string ids 0, 1, 2, ..."""
    ids = [memvid.put(title=f"doc-{i}", label="t", text=f"body {i}") for i in range(3)]
    assert ids == ["0", "1", "2"]


def test_put_many_returns_ids_for_all_requests(memvid: memvid_sdk.Memvid) -> None:
    """``put_many()`` returns one frame id per input, in order."""
    requests: List[Dict[str, Any]] = [
        {"title": "A", "label": "batch", "text": "aaa"},
        {"title": "B", "label": "batch", "text": "bbb"},
        {"title": "C", "label": "batch", "text": "ccc"},
    ]
    ids = memvid.put_many(requests)
    assert isinstance(ids, list)
    assert len(ids) == len(requests)
    assert all(isinstance(i, str) and i.isdigit() for i in ids)


def test_put_advances_frame_count(memvid: memvid_sdk.Memvid) -> None:
    """``stats()["frame_count"]`` advances by one per ``put()``."""
    before = memvid.stats()["frame_count"]
    memvid.put(title="x", label="y", text="hello")
    after = memvid.stats()["frame_count"]
    assert after == before + 1


def test_put_persists_metadata_and_tags(memvid: memvid_sdk.Memvid) -> None:
    """Metadata and tags attached at ``put()`` time survive round-trip to find."""
    memvid.put(
        title="Tagged",
        label="demo",
        text="unique-token xyzzy",
        tags=["alpha-tag", "beta-tag"],
    )
    memvid.seal()
    hits = memvid.find("xyzzy")["hits"]
    assert len(hits) == 1
    stored_tags: List[str] = hits[0]["tags"]
    assert "alpha-tag" in stored_tags
    assert "beta-tag" in stored_tags


# ---------------------------------------------------------------------------
# Search: find
# ---------------------------------------------------------------------------


def test_find_result_has_expected_top_level_keys(seeded_memvid: memvid_sdk.Memvid) -> None:
    """The ``find()`` result dict carries the documented envelope keys."""
    result = seeded_memvid.find("alpha")
    required: Set[str] = {"query", "hits", "total_hits", "took_ms", "engine"}
    missing = required - set(result.keys())
    assert not missing, f"find() result missing keys: {sorted(missing)}"
    assert result["query"] == "alpha"


def test_find_hit_has_expected_fields(seeded_memvid: memvid_sdk.Memvid) -> None:
    """Each hit exposes ``frame_id``, ``title``, ``rank``, ``score`` and ``snippet``."""
    hits: List[Dict[str, Any]] = seeded_memvid.find("alpha")["hits"]
    assert hits, "expected at least one hit for query 'alpha'"
    required: Set[str] = {"frame_id", "title", "rank", "score", "snippet", "uri"}
    for hit in hits:
        missing = required - set(hit.keys())
        assert not missing, f"hit missing keys: {sorted(missing)} (hit={hit})"


def test_find_no_match_returns_empty_hits(seeded_memvid: memvid_sdk.Memvid) -> None:
    """A query with no matches returns an empty ``hits`` list (not a missing key)."""
    result = seeded_memvid.find("nothingmatchesthisuniquewordzzz")
    assert "hits" in result
    assert result["hits"] == []
    assert result.get("total_hits", 0) == 0


def test_find_k_limits_results(seeded_memvid: memvid_sdk.Memvid) -> None:
    """``k=1`` caps the hits list to a single entry."""
    result = seeded_memvid.find("alpha", k=1)
    assert len(result["hits"]) == 1


def test_find_hits_ranked_by_score_descending(seeded_memvid: memvid_sdk.Memvid) -> None:
    """Hits are returned ordered by ``score`` descending, with ``rank`` 1..N."""
    hits = seeded_memvid.find("gamma")["hits"]
    assert len(hits) >= 2, "need at least two hits to check ordering"

    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True), f"scores not descending: {scores}"

    ranks = [h["rank"] for h in hits]
    assert ranks == list(range(1, len(hits) + 1)), f"ranks not 1..N: {ranks}"


def test_find_top_hit_is_most_relevant_document(seeded_memvid: memvid_sdk.Memvid) -> None:
    """For ``gamma``, the 4x-gamma frame 2 outranks the 1x-gamma frame 0."""
    hits = seeded_memvid.find("gamma")["hits"]
    assert int(hits[0]["frame_id"]) == 2, (
        f"expected frame 2 as top hit, got {hits[0]['frame_id']} (hits={hits})"
    )


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def test_timeline_contains_one_entry_per_put(seeded_memvid: memvid_sdk.Memvid) -> None:
    """The timeline has exactly one entry per ``put()`` call in the seed set."""
    entries = seeded_memvid.timeline()
    assert len(entries) == 3
    frame_ids = [int(e["frame_id"]) for e in entries]
    assert sorted(frame_ids) == [0, 1, 2]


def test_timeline_limit_caps_entries(seeded_memvid: memvid_sdk.Memvid) -> None:
    """The ``limit`` parameter caps the number of timeline entries returned."""
    entries = seeded_memvid.timeline(limit=2)
    assert len(entries) == 2


def test_timeline_reverse_flips_ordering(seeded_memvid: memvid_sdk.Memvid) -> None:
    """``reverse=True`` puts the newest frame first."""
    forward = seeded_memvid.timeline()
    reverse = seeded_memvid.timeline(reverse=True)
    assert int(forward[0]["frame_id"]) == 0
    assert int(reverse[0]["frame_id"]) == 2


# ---------------------------------------------------------------------------
# Memory cards
# ---------------------------------------------------------------------------


def test_add_memory_cards_reports_added_count(memvid: memvid_sdk.Memvid) -> None:
    """``add_memory_cards()`` reports the number of cards persisted."""
    result = memvid.add_memory_cards(
        [
            {"entity": "alice", "slot": "role", "value": "engineer"},
            {"entity": "alice", "slot": "team", "value": "platform"},
        ]
    )
    assert result["added"] == 2
    assert isinstance(result["ids"], list) and len(result["ids"]) == 2


def test_memories_returns_added_cards(memvid: memvid_sdk.Memvid) -> None:
    """``memories()`` returns the cards that were added."""
    memvid.add_memory_cards(
        [{"entity": "bob", "slot": "color", "value": "blue"}]
    )
    got = memvid.memories()
    assert got["count"] == 1
    card = got["cards"][0]
    assert card["entity"] == "bob"
    assert card["slot"] == "color"
    assert card["value"] == "blue"


def test_memory_entities_lists_distinct_entities(memvid: memvid_sdk.Memvid) -> None:
    """``memory_entities()`` lists each distinct ``entity`` exactly once."""
    memvid.add_memory_cards(
        [
            {"entity": "alice", "slot": "role", "value": "engineer"},
            {"entity": "bob", "slot": "role", "value": "designer"},
            {"entity": "alice", "slot": "team", "value": "platform"},
        ]
    )
    entities = set(memvid.memory_entities())
    assert entities == {"alice", "bob"}


def test_memories_stats_reflects_card_count(memvid: memvid_sdk.Memvid) -> None:
    """``memories_stats()`` reports the correct ``card_count``/``entity_count``."""
    memvid.add_memory_cards(
        [
            {"entity": "alice", "slot": "role", "value": "engineer"},
            {"entity": "bob", "slot": "role", "value": "designer"},
        ]
    )
    stats = memvid.memories_stats()
    assert stats["card_count"] == 2
    assert stats["entity_count"] == 2


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_session_start_returns_uuid_string(memvid: memvid_sdk.Memvid) -> None:
    """``session_start()`` returns a valid UUID string.

    Validation goes through ``uuid.UUID`` rather than pattern-matching
    the canonical textual form, so any standard UUID variant (v4, v7,
    no-hyphen, braced) is accepted — the SDK only promises "a UUID".
    """
    session_id = memvid.session_start(name="test-session")
    try:
        assert isinstance(session_id, str)
        uuid.UUID(session_id)  # raises ValueError if not a valid UUID string
    finally:
        memvid.session_end()


def test_session_end_returns_session_metadata(memvid: memvid_sdk.Memvid) -> None:
    """``session_end()`` returns a dict describing the ended session."""
    session_id = memvid.session_start(name="metadata-probe")
    memvid.put(title="in-session", label="s", text="activity during session")
    memvid.session_checkpoint()
    info = memvid.session_end()

    assert info["session_id"] == session_id
    assert info["name"] == "metadata-probe"
    assert info["action_count"] >= 1
    assert info["checkpoint_count"] >= 1
    assert "ended_secs" in info


def test_session_list_includes_ended_session(memvid: memvid_sdk.Memvid) -> None:
    """A session appears in ``session_list()`` after it has been ended."""
    sid = memvid.session_start(name="list-me")
    memvid.session_end()

    listed = memvid.session_list()
    assert any(s["session_id"] == sid for s in listed), (
        f"session {sid} not in {listed}"
    )


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


def test_remove_returns_nonnegative_int(memvid: memvid_sdk.Memvid) -> None:
    """``remove()`` returns a non-negative int (bytes reclaimed)."""
    fid = memvid.put(title="doomed", label="x", text="please delete me")
    memvid.seal()
    reclaimed = memvid.remove(fid)
    assert isinstance(reclaimed, int)
    assert reclaimed >= 0


def test_remove_prevents_future_find_hit(memvid: memvid_sdk.Memvid) -> None:
    """After ``remove()``, the frame no longer appears in ``find()`` results."""
    marker_text = "unique-marker-qqqqzzzz"
    fid = memvid.put(title="doomed", label="x", text=marker_text)
    memvid.seal()
    assert memvid.find(marker_text)["hits"], "precondition: find should hit before remove"

    memvid.remove(fid)
    memvid.seal()
    hits = memvid.find(marker_text)["hits"]
    assert all(int(h["frame_id"]) != int(fid) for h in hits), (
        f"removed frame {fid} still appears in hits: {hits}"
    )


# ---------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------


def test_seal_and_commit_do_not_raise(memvid: memvid_sdk.Memvid) -> None:
    """``seal()`` and ``commit()`` run without raising on a populated memory."""
    memvid.put(title="x", label="y", text="hello")
    memvid.seal()
    memvid.commit()


def test_reopen_read_only_preserves_puts(mv2_path: Path) -> None:
    """Writes survive ``close()`` + reopen in read-only mode."""
    writer = memvid_sdk.create(str(mv2_path), enable_lex=True)
    try:
        writer.put(title="persisted", label="x", text="survive-close-and-reopen")
        writer.seal()
    finally:
        writer.close()

    reader = memvid_sdk.use("basic", str(mv2_path), mode="open", read_only=True)
    try:
        assert reader.stats()["frame_count"] == 1
        hits = reader.find("survive-close-and-reopen")["hits"]
        assert len(hits) == 1
        assert hits[0]["title"] == "persisted"
    finally:
        reader.close()


def test_operation_on_closed_handle_raises_runtime_error(memvid: memvid_sdk.Memvid) -> None:
    """Calling methods after ``close()`` raises ``RuntimeError`` with a clear message.

    The Rust wrapper raises ``PyRuntimeError("memvid handle is closed")``
    (see ``bindings/python/src/lib.rs``). Matching ``r"handle is closed"``
    is specific enough to fail if the message drifts to an unrelated
    close-adjacent error.
    """
    memvid.close()

    with pytest.raises(RuntimeError, match=r"handle is closed"):
        memvid.stats()

    with pytest.raises(RuntimeError, match=r"handle is closed"):
        memvid.find("anything")


def test_closed_attribute_reflects_close(memvid: memvid_sdk.Memvid) -> None:
    """The ``closed`` attribute flips from ``False`` to ``True`` after ``close()``."""
    assert memvid.closed is False
    memvid.close()
    assert memvid.closed is True


# ---------------------------------------------------------------------------
# Integrity: verify, doctor, export_facts
# ---------------------------------------------------------------------------


def test_verify_reports_all_checks_passed_on_fresh_file(seeded_memvid: memvid_sdk.Memvid) -> None:
    """Every ``verify()`` check on a seeded, sealed memory has ``status == 'passed'``."""
    report = seeded_memvid.verify()
    assert isinstance(report, dict)
    assert "checks" in report
    assert report["checks"], "verify() returned no checks"

    failed = [c for c in report["checks"] if c.get("status") != "passed"]
    assert not failed, f"verify found failing checks: {failed}"


def test_doctor_dry_run_returns_findings_dict(seeded_memvid: memvid_sdk.Memvid) -> None:
    """``doctor(dry_run=True)`` returns a dict with a ``findings`` list."""
    report = seeded_memvid.doctor(dry_run=True, quiet=True)
    assert isinstance(report, dict)
    assert "findings" in report
    assert isinstance(report["findings"], list)


def test_export_facts_returns_valid_json(seeded_memvid: memvid_sdk.Memvid) -> None:
    """``export_facts(format='json')`` returns a string that parses as JSON."""
    dumped = seeded_memvid.export_facts(format="json")
    assert isinstance(dumped, str)
    parsed = json.loads(dumped)
    assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# Module-level: configure, validate, locking
# ---------------------------------------------------------------------------


def test_configure_and_get_config_round_trip() -> None:
    """``configure()`` stores values and ``get_config()`` returns them."""
    memvid_sdk.configure({"api_key": "mv2_test_roundtrip_value"})
    got = memvid_sdk.get_config()
    assert got.get("api_key") == "mv2_test_roundtrip_value"


def test_reset_config_clears_all_values() -> None:
    """``reset_config()`` wipes any previously-configured values."""
    memvid_sdk.configure({"api_key": "mv2_will_be_cleared"})
    assert memvid_sdk.get_config().get("api_key")
    memvid_sdk.reset_config()
    assert memvid_sdk.get_config() == {}


def test_validate_config_returns_expected_envelope() -> None:
    """``validate_config()`` returns a dict with ``memvid``, ``dashboard``, ``all_valid``."""
    report = memvid_sdk.validate_config()
    assert isinstance(report, dict)
    for key in ("memvid", "dashboard", "all_valid"):
        assert key in report, f"validate_config() missing key: {key}"
    assert isinstance(report["memvid"], dict)
    assert "configured" in report["memvid"]
    assert "valid" in report["memvid"]


def test_lock_who_reports_unlocked_on_idle_file(mv2_path: Path) -> None:
    """A file with no active handle reports ``locked == False``."""
    memvid_sdk.create(str(mv2_path)).close()
    status = memvid_sdk.lock_who(str(mv2_path))
    assert status["locked"] is False
    assert status["owner"] is None


def test_lock_nudge_returns_false_on_unlocked_file(mv2_path: Path) -> None:
    """``lock_nudge()`` returns ``False`` when there is nothing to nudge."""
    memvid_sdk.create(str(mv2_path)).close()
    assert memvid_sdk.lock_nudge(str(mv2_path)) is False


def test_verify_single_file_runs_on_valid_file(mv2_path: Path) -> None:
    """``verify_single_file()`` accepts a freshly created file without raising."""
    memvid_sdk.create(str(mv2_path)).close()
    memvid_sdk.verify_single_file(str(mv2_path))


# ---------------------------------------------------------------------------
# Free-tier file-size limit (stress, ~5 min)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_put_loop_grows_file_past_old_free_tier_ceiling(
    mv2_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Repeated ``put()`` can grow the ``.mv2`` file past 50 MB without an API key.

    Validates the lift of the free-tier file-size ceiling from the old
    50 MB cap (``FREE_TIER_LIMIT_BYTES`` in
    ``bindings/python/src/lib.rs``) to the new 9999 GB cap. With no
    ``MEMVID_API_KEY`` set, ``put()`` must now accept writes well past
    the old 50 MB boundary; this test drives the file to
    ``STRESS_TARGET_FILE_SIZE_BYTES`` (60 MB, 1.2x the old ceiling) in
    a loop of ~8 KiB ``put()`` calls and asserts every write succeeds.

    Failure modes this catches:

    * the old 50 MB ceiling (or any ceiling below the target) resurfaces
      anywhere in the write path — manifests as
      ``CapacityExceededError`` before the file reaches the target,
      and the handler below converts that into an explicit
      "free-tier ceiling regression" ``pytest.fail``, or
    * a regression in per-put payload growth — manifests as the loop
      exhausting its safety cap without reaching the target on disk.

    The goal here is **file-size growth**, not indexing correctness, so
    the handle is created with ``enable_lex=False, enable_vec=False``
    and every ``put()`` disables embedding, auto-tagging, and date
    extraction. Those defaults dominate per-put cost — turning them
    off keeps the small-chunk loop from bottlenecking on tokenization
    / regex work.

    Progress is printed every few iterations and at the end. Run with
    ``pytest -s`` (or ``--capture=no``) to see it live; otherwise
    pytest still shows the output on failure.

    Marked ``slow`` because it writes ~60 MB of random data across
    thousands of puts and takes ~5 min on average; deselect with
    ``pytest -m 'not slow'`` when running the fast smoke suite.
    """
    assert os.environ.get("MEMVID_API_KEY") is None, (
        "autouse fixture must have cleared MEMVID_API_KEY"
    )

    # Bypass the shared ``memvid`` fixture, which creates with
    # ``enable_lex=True`` and would then do expensive tokenization
    # bookkeeping on every put.
    mv = memvid_sdk.create(str(mv2_path), enable_lex=False, enable_vec=False)

    # ~8 KiB of url-safe base64 text per put. ``secrets.token_urlsafe(n)``
    # takes ``n`` entropy bytes and returns a base64url string of
    # length ~4/3 * n, so 6 KiB of entropy yields ~8 KiB of text.
    # Small chunks are deliberate: per-put cost in the write path
    # scales super-linearly with payload size on large single puts,
    # so many small puts reach the 60 MB target far faster than a few
    # big ones while still producing a realistic "many items" load.
    # Expected iterations: ~7,500 puts to reach 60 MB.
    entropy_bytes_per_put: int = 6 * 1024
    safety_cap: int = 15_000  # ~2x expected iters; guards against regressions
    items_put: int = 0
    file_size: int = 0

    # All per-put processing the high-level wrapper exposes is turned
    # off: this is a file-size stress test, not an indexing /
    # tagging / date-extraction test. (``enable_enrichment`` is not
    # forwarded by ``Memvid.put()`` — the rules engine still runs,
    # but on random base64 text it has nothing to match and stays
    # cheap.)
    put_opts: Dict[str, bool] = {
        "enable_embedding": False,
        "auto_tag": False,
        "extract_dates": False,
    }

    progress_every: int = 10  # print progress every N puts
    target_mb: float = STRESS_TARGET_FILE_SIZE_BYTES / (1 << 20)
    approx_text_bytes_per_put = entropy_bytes_per_put * 4 // 3
    t_start = time.perf_counter()

    # Disable pytest output capture for this block so the progress
    # prints show up live regardless of whether ``-s`` was passed.
    with capsys.disabled():
        print(
            f"\n[stress] target={target_mb:.0f} MB, "
            f"~{approx_text_bytes_per_put / 1024:.1f} KiB/put, "
            f"cap={safety_cap} iters",
            flush=True,
        )

        try:
            for i in range(safety_cap):
                payload = secrets.token_urlsafe(entropy_bytes_per_put)
                try:
                    mv.put(title=f"doc-{i}", label="stress", text=payload, **put_opts)
                except memvid_sdk.CapacityExceededError as exc:
                    pytest.fail(
                        f"free-tier ceiling regression: put() raised "
                        f"CapacityExceededError at file_size={file_size}, "
                        f"items_put={items_put} — a ceiling below "
                        f"{target_mb:.0f} MB has resurfaced on the write "
                        f"path. ({exc})"
                    )
                items_put += 1

                # Stat only when about to log or check the limit — one
                # ``stat()`` per put for 7.5k iterations is wasteful
                # overhead next to the put itself.
                if items_put == 1 or items_put % progress_every == 0:
                    file_size = mv2_path.stat().st_size
                    elapsed = time.perf_counter() - t_start
                    file_mb = file_size / (1 << 20)
                    pct = 100.0 * file_mb / target_mb
                    rate = items_put / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[stress] put {items_put:5d}  "
                        f"file={file_mb:6.2f} MB ({pct:5.1f}%)  "
                        f"elapsed={elapsed:6.2f}s  "
                        f"rate={rate:6.1f} put/s",
                        flush=True,
                    )
                    if file_size >= STRESS_TARGET_FILE_SIZE_BYTES:
                        break
            else:
                pytest.fail(
                    f"put() loop finished {safety_cap} iterations without "
                    f"reaching {STRESS_TARGET_FILE_SIZE_BYTES} bytes: final "
                    f"file_size={file_size}, items_put={items_put}. Per-put "
                    f"growth likely regressed — raise ``safety_cap`` or "
                    f"inspect payload encoding."
                )

            # Snapshot frame_count before ``seal()`` so the assertion
            # stays robust if ``seal()`` ever writes internal
            # housekeeping frames.
            pre_seal_frame_count: int = mv.stats()["frame_count"]

            # ``seal()`` flushes pending TOC writes so a final size
            # check reflects the fully-durable file rather than a
            # mid-flight state.
            mv.seal()
            final_file_size = mv2_path.stat().st_size

            total_elapsed = time.perf_counter() - t_start
            print(
                f"[stress] done: {items_put} puts, "
                f"file={final_file_size / (1 << 20):.2f} MB, "
                f"elapsed={total_elapsed:.2f}s",
                flush=True,
            )
        finally:
            mv.close()

    assert items_put > 0, "no puts succeeded"
    assert final_file_size >= STRESS_TARGET_FILE_SIZE_BYTES, (
        f"final file size {final_file_size} is below the "
        f"{STRESS_TARGET_FILE_SIZE_BYTES}-byte target after {items_put} puts"
    )
