# memvid-sdk-free (Python) — Production Readiness Status

This file tracks **CLI/core parity work** for the Python bindings (`proprietary/memvid-bindings/python`).

- **Source of truth**: repo root `enhancements.md`
- **Goal**: “Ingest anywhere, query anywhere” with loud, actionable errors (no silent wrong results).
- **Rule**: every item is “Done” only when **API + native wiring + tests + docs** are complete.

## Legend

- `TODO` = not implemented yet
- `PARTIAL` = implemented but not CLI/core parity (missing options / incorrect defaults / no tests)
- `DONE` = parity + tests + docs

---

## Phase 0 — Lifecycle / Introspection / Maintenance

### P0.0 Test harness + release gates (foundation)

- [x] **DONE** Add `pytest`-based tests + release gate command.
  - Command: `./.venv/bin/python -m pytest -q` (offline-safe by default)
  - Files: `pyproject.toml`, `tests/test_phase0.py`
- [x] **DONE** Add a `test:examples` runner that smoke-runs offline-safe scripts under `examples/` and skips network ones unless keys exist.
  - Command: `./.venv/bin/python scripts/test_examples.py`
  - Files: `scripts/test_examples.py`
- [x] **DONE** Add “golden fixture” tests (CLI↔Python smoke).
  - Python writes `.mv2` → CLI `find/ask` works (lex mode, offline)
  - Gated by `MEMVID_CLI_PATH` or `memvid` on PATH
  - Files: `tests/test_phase0.py`

### P0.1 `use()` / open/create semantics (correctness + DX)

- [x] **DONE** Ensure `use()` conventions match docs and expose core/CLI open flags.
  - Added: `lock_timeout_ms`, `force="stale_only"`, `force_writable`
  - Added: `lock_who()` / `lock_nudge()` helpers for lock diagnostics
  - Files: `memvid_sdk/__init__.py`, `README.md`, `src/lib.rs`
  - Tests: `tests/test_phase0.py`

### P0.2 `stats()` parity (must surface embedding identity + vec dimension)

- [x] **DONE** Extend Python `stats()` to include:
  - `effective_vec_dimension` (segment-aware)
  - `embedding_identity_summary` (unknown/single/mixed + counts)
  - Files: `src/lib.rs`
  - Tests: `tests/test_phase0.py`

### P0.3 Error mapping + stable error codes (no surprise exceptions)

- [x] **DONE** Map core errors to stable Python exception types + `MemvidError.code`.
  - Includes: `MV010` frame not found, `MV011` vec disabled, `MV012` corrupt file, `MV013` file not found, `MV014` dimension mismatch, `MV015` embedding failed, `MV016` clip disabled, `MV017` NER model unavailable
  - Files: `src/lib.rs`, `memvid_sdk/__init__.py`, `README.md`
  - Tests: `tests/test_phase0.py`

### P0.4 `verify` parity

- [x] **DONE** Provide instance-friendly `verify()` while keeping `Memvid.verify(path)` working.
  - Files: `memvid_sdk/__init__.py`, `src/lib.rs`
  - Tests: `tests/test_phase0.py`

### P0.5 `doctor` parity

- [x] **DONE** Expand `doctor()` options to match CLI and support plan-only (`dry_run=True`).
  - Files: `memvid_sdk/__init__.py`, `src/lib.rs`
  - Tests: `tests/test_phase0.py`

### P0.6 Single-file guarantee (`verify-single-file`)

- [x] **DONE** Add `verify_single_file(path)` helper (TS parity).
  - Files: `memvid_sdk/__init__.py`
  - Tests: `tests/test_phase0.py`

### P0.8 Version / diagnostics

- [x] **DONE** Add `info()` (Python) + `version_info()` (native) diagnostics.
  - Files: `memvid_sdk/__init__.py`, `src/lib.rs`
  - Tests: `tests/test_phase0.py`

### P0.9 Documentation correctness (must match shipped APIs)

- [x] **DONE** Keep README + examples in sync via tests + example smoke-run.
  - Files: `README.md`, `tests/test_phase0.py`, `scripts/test_examples.py`

### P0.10 External LLM provider: NVIDIA

- [x] **DONE** Support NVIDIA Integrate as an external LLM provider for answer synthesis.
  - Usage: `model="nvidia:<model>"` (or `model="nvidia"` with `NVIDIA_LLM_MODEL`)
  - Env: `NVIDIA_API_KEY` (+ optional `NVIDIA_BASE_URL`, `NVIDIA_LLM_MODEL`)
  - Files: `memvid/crates/memvid-ask-model/src/lib.rs`, `tests/test_llm_nvidia.py`
  - Tests:
    - offline-safe: `nvidia` without a model fails loudly (no network)
    - network-gated: `MEMVID_RUN_NVIDIA_LLM_TESTS=1` runs a real `ask()` synthesis call

---

## Phase 1 — Embeddings / Semantic Parity (start)

### P1.0 Persist embedding identity for precomputed embeddings

- [x] **DONE** Add `embedding_identity` support for `put_many(..., embeddings=...)` and persist into per-frame `extra_metadata` (`memvid.embedding.*`).
  - Fixes: CLI/SDK auto-detection across ingestion/query surfaces.
  - Files: `memvid_sdk/__init__.py`, `src/lib.rs`, `README.md`
  - Tests: `tests/test_find_semantic.py`

### P1.2 Semantic query support in SDK (parity with CLI)

- [x] **DONE** Full semantic `find/ask` parity using **pre-computed query embeddings** (offline-safe):
  - `find(mode="sem", query_embedding)` → pure vector search (with adaptive options).
  - `find(mode="auto", query_embedding)` → lexical search + semantic rerank (RRF).
  - `ask(mode="sem"/"auto", query_embedding)` → semantic/hybrid retrieval using native `askWithEmbedding`.
  - Adaptive retrieval flags surfaced: `adaptive`, `min_relevancy`, `max_k`, `adaptive_strategy`.
  - Default behavior must be truthful: semantic modes without a way to embed must fail loudly.
  - Files: `memvid_sdk/__init__.py`, `src/lib.rs`, `README.md`
  - Tests: `tests/test_find_semantic.py`, `tests/test_ask_semantic.py`

### P1.4 Auto query embeddings (model auto-detect + local/OpenAI runtime)

- [x] **DONE** Allow semantic `find/ask` without `query_embedding`:
  - Auto-detect embedding model from `.mv2` (`memvid.embedding.*` identity metadata, else vec dimension).
  - Embed query via runtime:
    - Local: `fastembed` (cache under `MEMVID_MODELS_DIR`)
    - OpenAI: `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`)
  - Error code: `MV015` when embeddings cannot be generated (missing key, offline, unknown model, mixed identity).
  - Files: `src/lib.rs`, `memvid_sdk/__init__.py`, `memvid_sdk/types.py`, `README.md`
  - Tests:
    - offline-safe: `tests/test_find_semantic.py`, `tests/test_ask_semantic.py`
    - cache/network-gated: `tests/test_runtime_embeddings_gated.py`

### P1.3 Truthful embedding flags (no misleading defaults)

- [x] **DONE** Make embedding-related defaults honest and CLI-compatible:
  - `put(..., enable_embedding=True)` must either:
    - store vectors (when vec enabled + embed runtime available), or
    - fail loudly with an actionable error code (no silent “lex-only” drift).
  - Support `embedding_model` naming aligned with CLI (`bge-small`, `openai-small`, etc).
  - Files: `memvid_sdk/__init__.py`, `src/lib.rs`, `README.md`
  - Tests: `tests/test_phase0.py`

### P1.5 External embedding providers (bring your own embedder)

- [x] **DONE** Promote `memvid_sdk.embeddings` into first-class SDK wiring:
  - `put_many(docs, { embedder })` fills missing `embeddings` and stores `embedding_identity`.
  - `find/ask({ mode:"sem"/"auto", embedder })` fills missing `query_embedding`.
  - Add a deterministic offline provider for tests (hash embedder), plus network-gated providers (OpenAI/Cohere/Voyage/NVIDIA/etc).
  - Files: `memvid_sdk/embeddings.py`, `memvid_sdk/__init__.py`, `src/lib.rs`, `README.md`
  - Tests: `tests/test_embedder_roundtrip.py`

---

## Phase 2 — Multimodal / Extraction (CLIP, NER, Tables)

### P2.0 Local CLIP support (native export + feature gating)

- [x] **DONE** Export `memvid_sdk._lib.ClipModel` and wire build features (offline-safe introspection + gated embedding tests).
  - Files: `pyproject.toml`, `Cargo.toml`, `src/lib.rs`, `memvid_sdk/clip.py`
  - Tests: `tests/test_phase2_models_tables.py` (export + dims, gated `MEMVID_TEST_CLIP=1` for real embeddings)

### P2.1 Local NER support (native export + feature gating)

- [x] **DONE** Export `memvid_sdk._lib.NerModel` and wire build features (deterministic missing-model error + gated extraction tests).
  - Files: `pyproject.toml`, `Cargo.toml`, `src/lib.rs`, `memvid_sdk/entities.py`
  - Tests: `tests/test_phase2_models_tables.py` (MV017 missing-model), gated `MEMVID_TEST_NER=1` for real extraction

### P2.2 Tables API parity

- [x] **DONE** Stabilize and test table ingestion APIs (`put_pdf_tables`, `list_tables`, `get_table`).
  - Fix: table storage returns real `frame_id`s (not WAL sequences) via `Memvid::next_frame_id()` (pending-insert aware), and `embed_rows` uses the SDK embedding runtime + persists embedding identity.
  - Files: `memvid/crates/memvid-core/src/table/storage.rs`, `memvid/crates/memvid-core/src/table/mod.rs`, `memvid_sdk/__init__.py`, `src/lib.rs`, `README.md`
  - Tests: `tests/test_phase2_models_tables.py`

---

## Notes / Constraints (non-negotiable for production)

- **Offline-first tests**: no network by default; network examples/tests must be gated by env vars.
- **No secrets in repo**: never hardcode provider keys; use env vars (`OPENAI_API_KEY`, etc).
- **Crate parity**: bindings should not silently drift from workspace `memvid-core` + `memvid-cli`.
