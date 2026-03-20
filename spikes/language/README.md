# Language Spike: Rust vs TypeScript/Bun

Architecture spike comparing Rust and TypeScript/Bun for Vera's key operations.

## Operations Tested

1. **Tree-sitter parsing** — Parse a ~8K LOC file (ripgrep flags/defs.rs)
2. **File tree traversal** — Walk turborepo (~5700 files) with gitignore awareness
3. **CLI cold start** — Time for a minimal binary/script to start and exit
4. **Binary/distribution size** — Single binary vs node_modules footprint

## Prerequisites

- Rust toolchain (cargo ≥1.75)
- Bun (≥1.0)
- Test repos cloned: `bash eval/setup-corpus.sh`

## Running

```bash
# Full benchmark suite
bash spikes/language/run-benchmarks.sh

# Individual Rust benchmarks
cd spikes/language/rust && cargo build --release
./target/release/spike-parse <file> [iterations]
./target/release/spike-walk <directory> [iterations]
./target/release/spike-startup

# Individual Bun benchmarks
cd spikes/language/ts-bun && bun install
bun run src/parse.ts <file> [iterations]
bun run src/walk.ts <directory> [iterations]
bun run src/startup.ts
```

## Results

See `results/benchmark-results.json` for raw data and `docs/adr/001-implementation-language.md` for the decision.
