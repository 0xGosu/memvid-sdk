#!/bin/bash
# setup-native.sh - Setup native Python bindings for macOS (Apple Silicon)
# Run this once to install, then use python3 directly for development

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Must run from proprietary/memvid-bindings/python directory"
    exit 1
fi

print_header "Native Python Bindings Setup for macOS"

# Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python version: $PYTHON_VERSION"

# Check for Rust
print_info "Checking Rust installation..."
if ! command -v rustc &> /dev/null; then
    print_error "Rust not found. Install from https://rustup.rs/"
    echo "Run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi
RUST_VERSION=$(rustc --version | awk '{print $2}')
print_success "Rust version: $RUST_VERSION"

# Check for maturin
print_info "Checking maturin..."
if ! command -v maturin &> /dev/null; then
    print_info "Installing maturin..."
    pip3 install maturin
fi
MATURIN_VERSION=$(maturin --version | awk '{print $2}')
print_success "Maturin version: $MATURIN_VERSION"

# Detect architecture
ARCH=$(uname -m)
print_info "Detected architecture: $ARCH"

if [ "$ARCH" == "arm64" ]; then
    TARGET="aarch64-apple-darwin"
    print_success "Building for Apple Silicon (ARM64)"
elif [ "$ARCH" == "x86_64" ]; then
    TARGET="x86_64-apple-darwin"
    print_success "Building for Intel (x86_64)"
else
    print_error "Unsupported architecture: $ARCH"
    exit 1
fi

# Install Rust target
print_info "Installing Rust target: $TARGET..."
rustup target add $TARGET
print_success "Target installed"

# Build memvid-core first (from workspace root)
#
# NOTE: `--all-features` enables CUDA (`memvid-core/cuda`) which requires `nvcc`.
# Most macOS machines don't have a CUDA toolkit, so we build the feature set the
# Python SDK actually uses (plus optional Whisper if requested).
print_header "Building memvid-core (SDK feature set)"
# Navigate to workspace root (memvid-projects)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
print_info "Building Rust core library..."

# Allow override from env for advanced users.
MEMVID_CORE_FEATURES="${MEMVID_CORE_FEATURES:-lex vec temporal_track parallel_segments clip logic_mesh replay}"

# Optional Whisper support (off by default: heavy build + optional model downloads at runtime).
if [ "${MEMVID_ENABLE_WHISPER:-0}" = "1" ]; then
    MEMVID_CORE_FEATURES="$MEMVID_CORE_FEATURES whisper"
    # Prefer Metal on macOS when Whisper is enabled.
    MEMVID_CORE_FEATURES="$MEMVID_CORE_FEATURES metal"
fi

print_info "memvid-core features: $MEMVID_CORE_FEATURES"
cargo build -p memvid-core --release --no-default-features --features "$MEMVID_CORE_FEATURES"
print_success "memvid-core built successfully"

# Back to bindings directory
cd "$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    print_header "Creating Virtual Environment"
    print_info "Creating .venv directory..."
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip and install maturin in venv
print_info "Installing/upgrading maturin in virtual environment..."
pip install --upgrade pip
pip install maturin
print_success "Maturin installed in virtual environment"

# Build and install Python bindings
print_header "Building Python Bindings (Native $TARGET)"
print_info "This will take 1-2 minutes on first build..."

# Use maturin develop for editable install (hot reload support)
maturin develop --release --target $TARGET

print_success "Python bindings installed!"

# Install development dependencies
print_header "Installing Development Dependencies"
if [ -f "requirements-dev.txt" ]; then
    print_info "Installing from requirements-dev.txt..."
    pip install -r requirements-dev.txt
    print_success "Development dependencies installed"
else
    print_warning "requirements-dev.txt not found, skipping"
fi

# Install Playwright browsers (required for browser automation)
print_header "Installing Playwright Browsers"
if pip show playwright &> /dev/null; then
    print_info "Installing Chromium browser for Playwright..."
    python -m playwright install chromium
    print_success "Playwright browsers installed"
else
    print_info "Playwright not installed, skipping browser installation"
fi

# Verify installation
print_header "Verifying Installation"
python -c "import memvid_sdk; print('✓ memvid_sdk imported successfully'); print(f'✓ Available: {dir(memvid_sdk)}')" 2>/dev/null && print_success "memvid_sdk is working!" || print_error "Import failed"

print_header "Setup Complete! 🎉"
echo ""
print_success "Virtual environment created at .venv/"
print_success "Python bindings installed in editable mode"
echo ""
print_warning "IMPORTANT: Activate the virtual environment before running scripts:"
echo "  source .venv/bin/activate"
echo ""
print_success "Then run Python scripts normally:"
echo "  python examples/existing_usage.py ../../../data/demo.mv2"
echo "  python examples/basic_usage.py"
echo ""
print_info "Development workflow:"
echo "  1. Activate venv: source .venv/bin/activate"
echo "  2. Edit Python code: memvid_sdk/*.py (changes are instant!)"
echo "  3. Edit Rust FFI: src/lib.rs (run this script again)"
echo "  4. Edit Rust core: ../../../memvid/crates/* (rebuild core + run this script)"
echo ""
print_info "To deactivate virtual environment: deactivate"
