# Architecture

Architectural decisions, patterns, and module responsibilities.

**What belongs here:** Module map, data flow, key types, architectural decisions.

---

## Workspace Crates

| Crate | Responsibility |
|-------|---------------|
| `vera-core` | Parsing, indexing, storage, embedding, retrieval, search pipeline |
| `vera-cli` | CLI interface (clap), user-facing commands |
| `vera-mcp` | MCP server (JSON-RPC stdio), tool definitions |
| `eval` | Evaluation harness, benchmark tasks, metrics |

## Key Modules in vera-core

- `parsing/` — Tree-sitter language grammars, AST chunking, symbol extraction
- `embedding/` — Embedding providers (API + local), dynamic dispatch
- `retrieval/` — Search pipeline: BM25 (Tantivy), vector (sqlite-vec), hybrid fusion, reranking
- `storage/` — SQLite index database, chunk storage
- `config.rs` — Runtime configuration, defaults
- `types.rs` — Language enum, SearchResult, SearchFilters, core types

## Data Flow

1. **Index**: Files → Tree-sitter parse → Symbol-aware chunks → Embeddings → SQLite + Tantivy
2. **Search**: Query → BM25 + Vector search → RRF fusion → Cross-encoder rerank → Ranked results
3. **Update**: Changed files (content hash) → Re-chunk + re-embed → Incremental update

## Tree-sitter Language Patterns

Three integration patterns exist:
1. **crates.io**: `tree-sitter-rust = "0.24"` in Cargo.toml
2. **git deps**: `tree-sitter-hcl = { git = "...", tag = "..." }` 
3. **in-repo C**: `crates/tree-sitter-sql/`, `crates/tree-sitter-proto/` with extern "C" bindings

## Tree-sitter Compatibility Notes

- New grammar crates must be compatible with the workspace `tree-sitter 0.26` ABI; older crates may compile research spikes but fail in the real workspace.
- Clojure currently uses `tree-sitter-clojure-orchard` because the more obvious `tree-sitter-clojure` package is not compatible with the workspace tree-sitter version.
- Some researched candidates were skipped during language expansion because viable grammars were not both maintained and 0.26-compatible (for example Nim/Hare ABI mismatches and an archived Hack grammar).
