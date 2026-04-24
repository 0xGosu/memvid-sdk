#!/usr/bin/env bash
# install_build_tools.sh
#
# Check for and install the build toolchain required to build memvid-sdk
# from source (Rust workspace + PyO3 bindings built with maturin).
#
# Idempotent: each step is skipped if the tool is already present.
#
# Tools handled:
#   - System packages (Linux/apt):  build-essential, pkg-config, libssl-dev, curl
#   - Rust toolchain (rustup, rustc, cargo)  via https://sh.rustup.rs
#   - maturin  (preferred: uv tool > pipx > pip --user)
#
# Usage:  bash scripts/install_build_tools.sh

set -euo pipefail

# ---------- pretty output ----------
if [ -t 1 ]; then
    C_RED='\033[0;31m'; C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'; C_BLUE='\033[0;34m'; C_OFF='\033[0m'
else
    C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''; C_OFF=''
fi
info()    { printf "${C_BLUE}[info]${C_OFF}  %s\n" "$*"; }
ok()      { printf "${C_GREEN}[ ok ]${C_OFF}  %s\n" "$*"; }
warn()    { printf "${C_YELLOW}[warn]${C_OFF}  %s\n" "$*"; }
err()     { printf "${C_RED}[err ]${C_OFF}  %s\n" "$*" >&2; }
section() { printf "\n${C_BLUE}== %s ==${C_OFF}\n" "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

# ---------- sudo helper ----------
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if have sudo; then
        SUDO="sudo"
    fi
fi

# ---------- 1. System packages ----------
section "System packages"

OS="$(uname -s)"
case "$OS" in
    Linux)
        if have apt-get; then
            REQUIRED_PKGS=(build-essential pkg-config libssl-dev curl ca-certificates)
            MISSING=()
            for pkg in "${REQUIRED_PKGS[@]}"; do
                if ! dpkg -s "$pkg" >/dev/null 2>&1; then
                    MISSING+=("$pkg")
                fi
            done
            if [ "${#MISSING[@]}" -eq 0 ]; then
                ok "All required apt packages already installed."
            else
                info "Installing: ${MISSING[*]}"
                if [ -z "$SUDO" ] && [ "$(id -u)" -ne 0 ]; then
                    err "Need root (sudo not available) to install: ${MISSING[*]}"
                    exit 1
                fi
                $SUDO apt-get update -y
                $SUDO apt-get install -y --no-install-recommends "${MISSING[@]}"
                ok "apt packages installed."
            fi
        else
            warn "Non-apt Linux detected. Please install equivalents of: build-essential pkg-config libssl-dev curl"
        fi
        ;;
    Darwin)
        if ! xcode-select -p >/dev/null 2>&1; then
            info "Installing Xcode Command Line Tools (interactive)..."
            xcode-select --install || true
        else
            ok "Xcode Command Line Tools present."
        fi
        if have brew; then
            for pkg in pkg-config openssl@3; do
                if brew list --formula "$pkg" >/dev/null 2>&1; then
                    ok "brew: $pkg already installed."
                else
                    info "brew install $pkg"
                    brew install "$pkg"
                fi
            done
        else
            warn "Homebrew not found. Install from https://brew.sh/ for pkg-config / openssl."
        fi
        ;;
    *)
        warn "Unsupported OS '$OS' — install build tools manually."
        ;;
esac

# ---------- 2. Rust toolchain ----------
section "Rust toolchain"

# Pick up an existing installation that isn't on PATH yet.
if ! have rustc && [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi

if have rustc && have cargo; then
    ok "Rust present: $(rustc --version)"
else
    info "Installing Rust via rustup (stable, default profile)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile default
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
    ok "Rust installed: $(rustc --version)"
    warn "Add Rust to your shell: source \"\$HOME/.cargo/env\""
fi

# ---------- 3. maturin ----------
section "maturin (PyO3 build backend)"

if have maturin; then
    ok "maturin present: $(maturin --version)"
else
    if have uv; then
        if uv tool list 2>/dev/null | grep -q '^maturin '; then
            ok "maturin already installed via uv tool (not on PATH in this shell)."
            warn "Ensure \$(uv tool dir --bin) is on PATH, or run: uv tool update-shell"
        else
            info "Installing maturin with: uv tool install maturin"
            uv tool install maturin
        fi
    elif have pipx; then
        if pipx list --short 2>/dev/null | grep -q '^maturin '; then
            ok "maturin already installed via pipx (not on PATH in this shell)."
            warn "Ensure ~/.local/bin is on PATH, or run: pipx ensurepath"
        else
            info "Installing maturin with: pipx install maturin"
            pipx install maturin
        fi
    elif have pip3 || have pip; then
        PIP_BIN="$(command -v pip3 || command -v pip)"
        info "Installing maturin with: $PIP_BIN install --user maturin"
        PIP_ERR="$(mktemp)"
        if "$PIP_BIN" install --user maturin 2>"$PIP_ERR"; then
            :
        elif grep -q 'externally-managed-environment' "$PIP_ERR"; then
            warn "System Python is externally-managed (PEP 668). Retrying with --break-system-packages."
            "$PIP_BIN" install --user --break-system-packages maturin
        else
            err "pip install failed:"
            cat "$PIP_ERR" >&2
            rm -f "$PIP_ERR"
            exit 1
        fi
        rm -f "$PIP_ERR"
        warn "Ensure ~/.local/bin is on PATH."
    else
        err "No Python package manager found (uv/pipx/pip). Install one and re-run."
        exit 1
    fi
    if have maturin; then
        ok "maturin installed: $(maturin --version)"
    else
        ok "maturin installed — start a new shell (or update PATH) to invoke it."
    fi
fi

# ---------- done ----------
section "Done"
ok "Build toolchain is ready."
cat <<EOF

Next steps to build the Python package from source:
  1) Create a venv:        uv venv .venv  (or: python3 -m venv .venv)
  2) Activate it:          source .venv/bin/activate
  3) Install maturin:      uv pip install maturin      # inside the venv
  4) Dev build + install:  cd bindings/python && maturin develop --release

See bindings/python/setup-native.sh for a full macOS dev setup.
EOF
