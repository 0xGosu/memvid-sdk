#!/usr/bin/env bash
# build_sdk.sh
#
# Build the memvid Python SDK (Rust extension via maturin) and run the
# basic test suite to verify the build.
#
# What this script does:
#   1. Verifies Rust is installed.
#   2. Creates/activates `.venv/` under the repo root.
#   3. Installs build dependencies (maturin + patchelf, pytest, onnxruntime).
#   4. Stages symlinks for the onnxruntime shared library in `.ort-lib/lib/`
#      so `ort-sys` can link against the copy that ships with the
#      `onnxruntime` Python wheel — this avoids a ~200MB download from
#      cdn.pyke.io which is blocked in some sandboxes.
#   5. Runs `maturin develop --release` with the feature set the SDK needs
#      (including the optional `fastembed` Rust feature so local embedding
#      models like bge-base work without an external API key).
#   6. Runs `tests/test_basic.py` to verify the installed extension works.
#   7. (Optional, with --wheel) Builds a distributable wheel into .wheels/
#      that can be installed into other venvs without re-running maturin.
#
# Usage:
#   bash scripts/build_sdk.sh                  # develop install + tests
#   bash scripts/build_sdk.sh --wheel          # also produce .wheels/*.whl
#   bash scripts/build_sdk.sh --wheel-out DIR  # custom wheel output dir

set -euo pipefail

# ---------- locate repo root ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------- CLI ----------
BUILD_WHEEL=0
WHEEL_OUT_DIR="${REPO_ROOT}/.wheels"

while [ $# -gt 0 ]; do
    case "$1" in
        --wheel|--with-wheel)
            BUILD_WHEEL=1
            shift
            ;;
        --wheel-out)
            BUILD_WHEEL=1
            WHEEL_OUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            printf "Unknown argument: %s\n" "$1" >&2
            exit 2
            ;;
    esac
done

# ---------- pretty output ----------
if [ -t 1 ]; then
    C_RED='\033[0;31m'; C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'; C_BLUE='\033[0;34m'; C_OFF='\033[0m'
else
    C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''; C_OFF=''
fi
log()     { printf "${C_BLUE}[build]${C_OFF}  %s\n" "$*"; }
ok()      { printf "${C_GREEN}[  ok ]${C_OFF}  %s\n" "$*"; }
warn()    { printf "${C_YELLOW}[warn ]${C_OFF}  %s\n" "$*"; }
err()     { printf "${C_RED}[error]${C_OFF}  %s\n" "$*" >&2; }
section() { printf "\n${C_BLUE}== %s ==${C_OFF}\n" "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

# ---------- 1. prerequisites ----------
section "Checking build prerequisites"

# Pick up an existing rustup installation that isn't on PATH yet.
if ! have rustc && [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi

if ! have rustc || ! have cargo; then
    err "Rust toolchain not found. Run: bash scripts/install_build_tools.sh"
    exit 1
fi
ok "Rust: $(rustc --version)"

if ! have python3; then
    err "python3 not found."
    exit 1
fi
ok "Python: $(python3 --version)"

# ---------- 2. virtual env ----------
section "Setting up Python virtual environment"

VENV_DIR="${REPO_ROOT}/.venv"
if [ ! -d "${VENV_DIR}" ]; then
    log "Creating venv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
ok "Activated venv at ${VENV_DIR}"

# ---------- 3. python build deps ----------
section "Installing Python build dependencies"

python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet \
    'maturin[patchelf]>=1.7' \
    'pytest>=7.0' \
    'onnxruntime'
ok "maturin: $(maturin --version)"
ok "pytest + onnxruntime installed in venv"

# ---------- 4. stage onnxruntime shared library ----------
section "Staging onnxruntime shared library for ort-sys"

ORT_STAGE_DIR="${REPO_ROOT}/.ort-lib/lib"
ORT_CAPI_DIR="$(python3 -c 'import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), "capi"))')"
if [ -z "${ORT_CAPI_DIR}" ] || [ ! -d "${ORT_CAPI_DIR}" ]; then
    err "Could not locate onnxruntime capi/ directory"
    exit 1
fi

# Pick the versioned libonnxruntime.so.<X.Y.Z> shipped with the wheel.
ORT_REAL_LIB="$(find "${ORT_CAPI_DIR}" -maxdepth 1 -name 'libonnxruntime.so.*' -print -quit)"
if [ -z "${ORT_REAL_LIB}" ]; then
    err "No libonnxruntime.so.* found in ${ORT_CAPI_DIR}"
    exit 1
fi

# Read the SONAME the linker actually needs (e.g. libonnxruntime.so.1)
# from the ELF header, falling back to a filename-based guess.
ORT_SONAME=""
if have readelf; then
    ORT_SONAME="$(readelf -d "${ORT_REAL_LIB}" 2>/dev/null \
        | awk '/SONAME/ { gsub(/[\[\]]/, "", $NF); print $NF; exit }')"
fi
if [ -z "${ORT_SONAME}" ]; then
    # Fallback: trim trailing .minor.patch from libonnxruntime.so.1.24.3
    ORT_SONAME="$(basename "${ORT_REAL_LIB}" | sed 's/\(libonnxruntime\.so\.[0-9][0-9]*\)\..*/\1/')"
fi

mkdir -p "${ORT_STAGE_DIR}"
ln -sf "${ORT_REAL_LIB}" "${ORT_STAGE_DIR}/libonnxruntime.so"
ln -sf "${ORT_REAL_LIB}" "${ORT_STAGE_DIR}/${ORT_SONAME}"
ln -sf "${ORT_REAL_LIB}" "${ORT_STAGE_DIR}/$(basename "${ORT_REAL_LIB}")"
ok "Staged symlinks in ${ORT_STAGE_DIR}:"
ls -l "${ORT_STAGE_DIR}" | sed 's/^/           /'

export ORT_LIB_LOCATION="${ORT_STAGE_DIR}"
export ORT_STRATEGY="system"
export ORT_SKIP_DOWNLOAD="1"
export ORT_PREFER_DYNAMIC_LINK="1"

# ---------- 5. build the extension ----------
section "Building memvid-sdk Python extension (maturin develop --release)"

# Features: everything the SDK ships with, including `fastembed` so local
# embedding models (bge-base, etc.) work out-of-the-box. The onnxruntime
# shared lib staged above satisfies fastembed's ort-sys link step.
MATURIN_FEATURES=(
    --no-default-features
    -F lex
    -F vec
    -F temporal_track
    -F parallel_segments
    -F clip
    -F logic_mesh
    -F replay
    -F encryption
    -F pdf_extract
    -F fastembed
)

maturin develop --release "${MATURIN_FEATURES[@]}"
ok "Extension built and installed in venv"

# ---------- 6. smoke import ----------
section "Smoke-testing import"

python3 -c "import memvid_sdk, memvid_sdk._lib; print('import ok:', memvid_sdk._lib.__file__)"
ok "memvid_sdk imports cleanly"

# ---------- 7. run basic tests ----------
section "Running tests/test_basic.py"

pytest tests/test_basic.py -v

# ---------- 8. optional distributable wheel ----------
# Reuses the warm cargo cache from the `maturin develop` above, so this only
# adds the wheel-packaging step (a few seconds), not a full Rust recompile.
if [ "${BUILD_WHEEL}" = "1" ]; then
    section "Building distributable wheel"
    mkdir -p "${WHEEL_OUT_DIR}"
    maturin build --release "${MATURIN_FEATURES[@]}" --out "${WHEEL_OUT_DIR}"

    WHEEL_PATH="$(ls -t "${WHEEL_OUT_DIR}"/memvid*.whl 2>/dev/null | head -1)"
    if [ -z "${WHEEL_PATH}" ]; then
        err "No wheel produced in ${WHEEL_OUT_DIR}"
        exit 1
    fi
    ok "Wheel: ${WHEEL_PATH}"
fi

# ---------- done ----------
section "Build complete"
ok "memvid-sdk-free installed in ${VENV_DIR}"
ok "Run more tests with:   source .venv/bin/activate && pytest tests/ -v"
if [ "${BUILD_WHEEL}" = "1" ]; then
    ok "Install the wheel elsewhere: pip install ${WHEEL_PATH}"
fi
