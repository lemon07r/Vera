#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load API credentials if available
if [ -f "$REPO_ROOT/secrets.env" ]; then
    set -a
    source "$REPO_ROOT/secrets.env"
    set +a
    echo "[init] Loaded secrets.env"
else
    echo "[init] WARNING: secrets.env not found - embedding/reranker APIs will not work"
fi

# Ensure Rust toolchain is available
if ! command -v cargo &>/dev/null; then
    echo "[init] ERROR: cargo not found. Install Rust: https://rustup.rs"
    exit 1
fi

echo "[init] Rust $(rustc --version)"

echo "[init] Downloading vendored grammars if needed..."
bash "$REPO_ROOT/scripts/bootstrap-vendored-grammars.sh"

# Build the project
if [ -f "$REPO_ROOT/Cargo.toml" ]; then
    echo "[init] Building project..."
    cargo build 2>&1 | tail -5
    echo "[init] Build complete"
fi

# Create benchmark repos directory
mkdir -p "$REPO_ROOT/.bench/repos"

echo "[init] Environment ready"
