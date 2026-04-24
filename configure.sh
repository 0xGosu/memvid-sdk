#!/bin/bash
set -e

echo "This script sets up all system config/files required to run the project"

# Run from the repo root regardless of the caller's CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# set up env files from default file
[ -f ".env" ] || cp default.env .env

# Install build toolchain (Rust, maturin, system deps).
bash "$SCRIPT_DIR/scripts/install_build_tools.sh"
