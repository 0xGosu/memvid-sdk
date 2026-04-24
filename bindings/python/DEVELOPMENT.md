# Python Bindings - Native Development Guide

## 🚀 Quick Start (First Time Setup)

Run the setup script once:

```bash
cd proprietary/memvid-bindings/python
./setup-native.sh
```

This will:
- ✅ Create a Python virtual environment (`.venv/`)
- ✅ Install Rust toolchain (if needed)
- ✅ Install maturin (Python-Rust build tool)
- ✅ Build `memvid-core` with the **SDK feature set** (lex, vec, parallel_segments, temporal_track, clip, logic_mesh)
- ✅ Build and install Python bindings for your native architecture (Apple Silicon or Intel)
- ✅ Install as editable package (hot reload for Python code changes!)
- ✅ Install all development dependencies (beautifulsoup4, browser-use, pytest, black, etc.)
- ✅ Install Playwright browsers (Chromium for webshot_agent.py)

**Time:** ~2-5 minutes on first run

**After setup, activate the virtual environment:**
```bash
source .venv/bin/activate
```

---

## 💻 Daily Development Workflow

### Activate Virtual Environment (First!)

**IMPORTANT:** Always activate the virtual environment before running scripts:

```bash
cd proprietary/memvid-bindings/python
source .venv/bin/activate
```

Your prompt will change to show `(.venv)` prefix.

### Run Python Scripts Directly

```bash
# Now just use python directly (no python3 needed!)
python examples/existing_usage.py ../../../data/demo.mv2

# Run basic usage example
python examples/basic_usage.py

# Interactive Python shell
python
>>> import memvid_sdk
>>> mem = memvid_sdk.Memvid.create("/tmp/test.mv2")
>>> mem.put("Hello, world!")
>>> mem.seal()

# When done, deactivate virtual environment
>>> exit()
deactivate
```

### Making Changes

**1. Edit Pure Python Code (INSTANT - No Rebuild!)**
```bash
# Edit Python modules
vim memvid_sdk/__init__.py

# Changes are immediate - just run your script
python3 examples/existing_usage.py demo.mv2
```

**2. Edit Rust FFI Code (Rebuild Bindings)**
```bash
# Edit Rust bindings
vim src/lib.rs

# Rebuild bindings only (fast ~10-30 seconds)
./setup-native.sh
```

**3. Edit Rust Core (Rebuild Core + Bindings)**
```bash
# Edit core library
vim ../../../memvid/crates/memvid-core/src/foo.rs

# Rebuild everything (~1-2 minutes)
# NOTE: Avoid `--all-features` on macOS unless you have CUDA (`nvcc`) installed.
# `--all-features` enables the CUDA backend via Candle → cudarc, which requires `nvcc`.
cd ../../../memvid
cargo build -p memvid-core --release --no-default-features --features "lex vec temporal_track parallel_segments clip logic_mesh"

cd ../proprietary/memvid-bindings/python
./setup-native.sh
```

---

## 📦 Dependencies

The setup script automatically installs:

**Core Dependencies:**
- `memvid_sdk` (compiled Rust extension)

**Development Dependencies** (from `requirements-dev.txt`):
- `beautifulsoup4` - HTML parsing (for webshot_agent.py)
- `readability-lxml` - Content extraction
- `lxml` - XML/HTML processing
- `browser-use` - High-level browser automation (for webshot_agent.py)
- `playwright` - Browser automation framework
- `pytest` - Testing framework
- `pytest-asyncio` - Async test support
- `black` - Code formatting
- `mypy` - Type checking
- `ruff` - Fast linter

To install additional dependencies:
```bash
source .venv/bin/activate
pip install package-name
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_basic.py -v

# Run with output
pytest tests/ -v -s
```

---

## 🔧 Architecture-Specific Builds

The setup script automatically detects your architecture:

- **Apple Silicon (M1/M2/M3)**: Builds for `aarch64-apple-darwin`
- **Intel Mac**: Builds for `x86_64-apple-darwin`

This **avoids universal2 build issues** that cause ARM linking errors.

---

## 📦 Building Wheels (Distribution)

For distributing the package:

```bash
# Build wheel for your architecture
maturin build --release --target $(uname -m)-apple-darwin

# Output: target/wheels/memvid_sdk-*.whl
```

For universal2 builds (both architectures):
```bash
# Requires Xcode command line tools
maturin build --release --target universal2-apple-darwin
```

---

## 🐛 Troubleshooting

### "No module named memvid_sdk"

**Solution:** Run `./setup-native.sh` to install the package

### "Symbol not found: _PyInit_memvid_sdk"

**Solution:** Architecture mismatch. Run `./setup-native.sh` which builds for your native architecture

### "rustc: command not found"

**Solution:** Install Rust from https://rustup.rs/
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### "maturin: command not found"

**Solution:** Install maturin
```bash
pip3 install maturin
```

### Linking errors on Apple Silicon

**Solution:** The setup script builds for `aarch64-apple-darwin` only (not universal2), which avoids these issues

---

## 📁 Project Structure

```
python/
├── setup-native.sh          # Setup script (run once)
├── DEVELOPMENT.md           # This file
├── pyproject.toml           # Python package config
├── Cargo.toml               # Rust FFI config
├── src/
│   └── lib.rs              # Rust FFI implementation
├── memvid_sdk/
│   ├── __init__.py         # Python API
│   └── _lib.*.so           # Compiled native extension
├── examples/
│   ├── basic_usage.py      # Basic example
│   └── existing_usage.py   # Advanced example
└── tests/
    └── test_*.py           # Test suite
```

---

## 🔗 Links

- **Memvid Core**: `../../../memvid/crates/memvid-core/`
- **CLI**: `../../../memvid/crates/memvid-cli/`
- **API Docs**: `../../../memvid/README.md`

---

## ✨ Features Enabled

The native build includes all features:
- ✅ `lex` - Full-text search (Tantivy)
- ✅ `vec` - Semantic search (HNSW + embeddings)
- ✅ `parallel_segments` - Multi-threaded ingestion
- ✅ `temporal_track` - Natural language temporal resolution

All features are available in the Python SDK!
