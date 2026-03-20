# ADR-001: Implementation Language

**Status:** Accepted  
**Date:** 2026-03-20

## Question

What implementation language should Vera use for its core CLI tool? Vera is a code indexing and retrieval system for AI coding agents, requiring fast tree-sitter parsing, efficient file system traversal, low CLI startup overhead, and compact distribution. The language choice affects performance, distribution model, ecosystem access, and long-term maintainability.

## Options Considered

### Option A: Rust

- **Tree-sitter:** Native C bindings via `tree-sitter` crate (v0.24). Zero-copy parsing. Mature ecosystem with grammar crates for all major languages.
- **File traversal:** `ignore` crate provides gitignore-aware, parallel directory walking. Used by ripgrep itself.
- **Distribution:** Single static binary via `cargo install` or release binaries. No runtime dependencies.
- **Ecosystem:** Strong crate ecosystem for CLI (clap), serialization (serde), async (tokio), BM25 (tantivy), vector search (lance, sqlite-vec). Well-suited for systems programming.
- **Maintenance risk:** Steeper learning curve for some contributors. Compile times moderate (~7s for this spike).

### Option B: TypeScript with Bun Runtime

- **Tree-sitter:** Node bindings via `tree-sitter` npm package (v0.22). Requires native compilation via node-gyp during install.
- **File traversal:** Bun built-in `Glob` API for fast scanning. Node `fs` for fallback. No gitignore-aware walker in standard library (would need custom implementation or third-party package).
- **Distribution:** `bun build --compile` for single binary (~60-80MB) or require Bun runtime installed. `node_modules` adds ~32MB for tree-sitter grammars alone.
- **Ecosystem:** Rich npm ecosystem. Faster iteration for prototyping. Strong JSON handling.
- **Maintenance risk:** Bun is relatively young (v1.3.x). Node-gyp native builds can be fragile across platforms. Bun's single-binary compilation is still maturing.

## Evaluation Method

Built minimal spike implementations in both languages measuring four key operations on identical inputs:

1. **Tree-sitter parse speed:** Parse + full AST walk of a ~8K LOC Rust file (ripgrep `flags/defs.rs`, 235KB), 20 iterations after warmup.
2. **File tree traversal:** Walk turborepo (~5700 files) with gitignore awareness, 10 iterations after warmup.
3. **CLI cold start time:** Spawn a minimal binary/script 100 times, measure average startup latency.
4. **Binary/distribution size:** Compare release binary size vs node_modules + runtime.

**Environment:** AMD Ryzen 5 7600X3D (12 threads), 30GB RAM, Arch Linux. Rust 1.94.0, Bun 1.3.11, Node 25.8.1.

**Spike code:** `spikes/language/rust/` and `spikes/language/ts-bun/`

## Evidence

### Tree-Sitter Parsing (parse + full AST node walk)

| File | LOC | Rust avg (ms) | Bun avg (ms) | Rust speedup |
|------|-----|---------------|---------------|--------------|
| ripgrep flags/defs.rs | 7,779 | 16.9 | 29.7 | **1.76×** |
| fastify hooks.test.js | 3,578 | 10.4 | 17.9 | **1.72×** |
| turborepo builder.rs | 4,945 | 14.0 | 22.8 | **1.63×** |

Rust is consistently **1.6–1.8× faster** at tree-sitter parsing. Both use the same underlying C tree-sitter library; the difference is FFI overhead and memory management (Rust's zero-copy vs Node binding's copying).

### File Tree Traversal

| Repo | Rust (ms) | Rust files | Bun (ms) | Bun files | Note |
|------|-----------|------------|----------|-----------|------|
| turborepo | 38.7 | 5,818 | 17.6 | 5,419 | Bun faster but walks fewer files (no dotfiles) |
| ripgrep | 1.4 | 247 | 0.7 | 209 | Both fast for small repos |

Bun's `Glob` API is faster in raw wall-clock time, but **walks fewer files** (skips dotfiles by default). The Rust `ignore` crate walks more thoroughly (hidden files included, full gitignore semantics). For Vera's use case, the `ignore` crate's gitignore-aware, configurable walking is the right abstraction — it's what ripgrep uses. Building equivalent gitignore support in TS/Bun would require significant custom code.

### CLI Cold Start

| Runtime | Avg startup (ms) | vs Rust |
|---------|-------------------|---------|
| **Rust** | **0.51** | — |
| Bun | 5.09 | 10× slower |
| Node | 35.52 | 70× slower |

Rust cold start is **10× faster than Bun** and **70× faster than Node**. For a CLI tool invoked frequently by agents (potentially hundreds of times per session), sub-millisecond startup is a significant advantage.

### Binary / Distribution Size

| Artifact | Size |
|----------|------|
| Rust spike-parse (with 3 grammars) | 2.6 MB |
| Rust spike-walk (with ignore/glob) | 3.1 MB |
| Rust spike-startup (minimal) | 444 KB |
| Estimated Vera full binary | **~10-15 MB** |
| TS/Bun node_modules (tree-sitter only) | 32 MB |
| Bun compiled binary (estimated) | ~60-80 MB |

Rust produces a **single ~10-15MB binary** (estimated for full Vera with all grammars and dependencies). The TS/Bun approach requires either the Bun runtime (~100MB) plus node_modules (32MB+), or a compiled Bun binary (~60-80MB). Rust's distribution story is significantly simpler and more compact.

### Ecosystem Assessment

| Capability | Rust | TypeScript/Bun |
|-----------|------|----------------|
| Tree-sitter bindings | ★★★ Native, mature | ★★ Node-gyp, works but fragile |
| Gitignore-aware walk | ★★★ `ignore` crate (ripgrep) | ★ No standard solution |
| BM25 full-text search | ★★★ Tantivy | ★★ Lunr/MiniSearch (less capable) |
| Vector operations | ★★★ Lance, sqlite-vec | ★★ Better-sqlite3 + extensions |
| CLI framework | ★★★ Clap | ★★★ Commander/yargs |
| JSON serialization | ★★★ Serde | ★★★ Native |
| Cross-platform builds | ★★★ `cross` for CI | ★★ Bun compile still maturing |
| MCP server | ★★ Possible, less common | ★★★ Native JSON-RPC/HTTP |

Rust has a clear edge in the core indexing/retrieval dependencies (tree-sitter, file walking, BM25, vector search). TypeScript has an edge for MCP server implementation, but this is a small fraction of the codebase.

## Decision

**Rust** is the chosen implementation language for Vera.

The decision is based on:
1. **1.6-1.8× faster tree-sitter parsing** — directly impacts index build time
2. **10× faster cold start** — critical for CLI tool invoked frequently by agents
3. **~5× smaller distribution** — single binary vs runtime + node_modules
4. **Superior ecosystem fit** — `ignore` crate, Tantivy, Lance are exactly the libraries Vera needs
5. **Simpler distribution** — `cargo install vera` produces one binary with zero runtime dependencies

## Consequences

**Gains:**
- Best-in-class performance for a CLI tool (sub-millisecond startup, fast parsing)
- Single binary distribution simplifies agent installation and updates
- Mature, proven ecosystem for all core dependencies (tree-sitter, file walking, BM25, vector search)
- Strong type system catches errors at compile time

**Trade-offs accepted:**
- Slower iteration speed during development compared to TypeScript
- Steeper learning curve for contributors unfamiliar with Rust
- MCP server implementation will require more boilerplate than a Node.js equivalent
- Compile times (~7-30s for incremental builds) vs instant TS startup

**Mitigations:**
- Use `cargo watch` for fast feedback during development
- Keep module boundaries clean and small to limit recompilation scope
- For MCP server, leverage `tower`/`axum` or similar HTTP framework with minimal boilerplate

## Follow-up

1. Validate that Rust's tree-sitter crate v0.24+ has stable multi-language support for Vera's Tier 1 languages
2. Confirm Tantivy BM25 performance in the storage backend spike (ADR-002)
3. Test cross-compilation for macOS (secondary platform) using `cross`
4. Monitor Bun's `--compile` maturity — if it stabilizes and Vera's next version needs rapid prototyping, TypeScript could serve as a plugin/extension language
