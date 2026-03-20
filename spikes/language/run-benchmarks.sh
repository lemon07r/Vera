#!/usr/bin/env bash
# Language spike benchmark runner
# Compares Rust vs TypeScript/Bun for Vera's key operations:
#   1. Tree-sitter parsing speed
#   2. File tree traversal
#   3. CLI cold start time
#   4. Binary size
#
# Prerequisites:
#   - Rust toolchain (cargo, rustc)
#   - Bun (bun)
#   - Test repos cloned in .bench/repos/ (run eval/setup-corpus.sh)
#
# Usage: bash spikes/language/run-benchmarks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

echo "================================================="
echo "  Vera Language Spike: Rust vs TypeScript/Bun"
echo "================================================="
echo ""

# --- Build Rust spike ---
echo "[1/6] Building Rust spike..."
cd "$SCRIPT_DIR/rust"
cargo build --release 2>&1 | tail -2
RUST_BIN="$SCRIPT_DIR/rust/target/release"
echo "  Done."
echo ""

# --- Install TS/Bun deps ---
echo "[2/6] Installing TS/Bun dependencies..."
cd "$SCRIPT_DIR/ts-bun"
bun install --silent 2>&1
echo "  Done."
echo ""

# --- Parse Benchmarks ---
TEST_FILE="$REPO_ROOT/.bench/repos/ripgrep/crates/core/flags/defs.rs"
PARSE_ITERS=20

if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: Test file not found: $TEST_FILE"
    echo "Run: bash eval/setup-corpus.sh"
    exit 1
fi

echo "[3/6] Parse benchmark: $(basename $TEST_FILE) (~7800 LOC, $PARSE_ITERS iterations)"
echo "  Rust:"
"$RUST_BIN/spike-parse" "$TEST_FILE" "$PARSE_ITERS" | tee "$RESULTS_DIR/parse-rust.json"
echo ""
echo "  Bun:"
cd "$SCRIPT_DIR/ts-bun"
bun run src/parse.ts "$TEST_FILE" "$PARSE_ITERS" | tee "$RESULTS_DIR/parse-bun.json"
echo ""

# --- Walk Benchmarks ---
WALK_DIR="$REPO_ROOT/.bench/repos/turborepo"
WALK_ITERS=10

if [ ! -d "$WALK_DIR" ]; then
    echo "ERROR: Walk directory not found: $WALK_DIR"
    exit 1
fi

echo "[4/6] Walk benchmark: turborepo (~5700 files, $WALK_ITERS iterations)"
echo "  Rust:"
"$RUST_BIN/spike-walk" "$WALK_DIR" "$WALK_ITERS" | tee "$RESULTS_DIR/walk-rust.json"
echo ""
echo "  Bun:"
cd "$SCRIPT_DIR/ts-bun"
bun run src/walk.ts "$WALK_DIR" "$WALK_ITERS" | tee "$RESULTS_DIR/walk-bun.json"
echo ""

# --- Cold Start ---
COLD_ITERS=100

echo "[5/6] Cold start benchmark ($COLD_ITERS invocations)"
echo "  Rust:"
START=$(date +%s%N)
for i in $(seq 1 $COLD_ITERS); do "$RUST_BIN/spike-startup" >/dev/null; done
END=$(date +%s%N)
RUST_MS=$(( (END - START) / 1000000 ))
RUST_AVG=$(python3 -c "print(f'{$RUST_MS/$COLD_ITERS:.3f}')")
echo "    Total: ${RUST_MS}ms, Avg: ${RUST_AVG}ms"

echo "  Bun:"
START=$(date +%s%N)
for i in $(seq 1 $COLD_ITERS); do bun run "$SCRIPT_DIR/ts-bun/src/startup.ts" >/dev/null 2>/dev/null; done
END=$(date +%s%N)
BUN_MS=$(( (END - START) / 1000000 ))
BUN_AVG=$(python3 -c "print(f'{$BUN_MS/$COLD_ITERS:.3f}')")
echo "    Total: ${BUN_MS}ms, Avg: ${BUN_AVG}ms"
echo ""

# --- Binary Size ---
echo "[6/6] Binary / distribution size"
echo "  Rust binaries:"
ls -lh "$RUST_BIN/spike-parse" "$RUST_BIN/spike-walk" "$RUST_BIN/spike-startup" 2>/dev/null | awk '{print "    " $5 " " $NF}'
echo "  TS/Bun node_modules:"
du -sh "$SCRIPT_DIR/ts-bun/node_modules/" | awk '{print "    " $1 " " $2}'
echo ""

echo "================================================="
echo "  Benchmarks complete. Results in $RESULTS_DIR/"
echo "================================================="
