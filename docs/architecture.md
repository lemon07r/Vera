# Architecture

## Workspace Crates

| Crate | Purpose | Entry point |
|-------|---------|-------------|
| `vera-core` | Parsing, indexing, storage, embedding, retrieval | `lib.rs` |
| `vera-cli` | CLI (clap derive macros) | `main.rs` |
| `vera-mcp` | MCP server (JSON-RPC over stdio) | `server.rs` |
| `eval` | Benchmark harness and metrics | `src/main.rs` |
| `tree-sitter-{sql,proto,vue,dockerfile,astro}` | Vendored C grammars (built via `cc`) | `build.rs` |

## vera-core modules

### `parsing/`: Language parsing & symbol extraction

Files: `mod.rs` (public API), `languages.rs` (grammar dispatch), `extractor.rs` (AST node → SymbolType), `chunker.rs` (symbol-aware, whole-file, and Tier 0 chunking).

Data flow: file → grammar lookup → tree-sitter parse → node classification → chunk production.

### `embedding/`: Embedding generation

`EmbeddingProvider` trait with two implementations:
- `ApiEmbeddingProvider`: HTTP calls to OpenAI-compatible endpoints
- `LocalEmbeddingProvider`: ONNX Runtime inference with Jina v5 nano

`DynamicProvider` dispatches between them at runtime based on `--local` flag.

### `retrieval/`: Search pipeline

1. Query enters `search_service.rs`
2. BM25 (`bm25.rs`) and vector search (`vector.rs`) run in parallel
3. Results fused via RRF (`hybrid.rs`, k=60). `fuse_rrf_multi` generalizes fusion to N ranked lists.
4. Query-aware ranking and candidate shaping apply deterministic priors (`ranking.rs`, `search_service.rs`)
5. Top candidates reranked by cross-encoder (`reranker.rs` or `local_reranker.rs`)
6. Final `Vec<SearchResult>` returned

Deep search (`--deep`): `rag_fusion.rs` expands the query into multiple variants via an LLM completion endpoint (`completion_client.rs`), runs parallel hybrid searches, and fuses results with N-way RRF. Falls back to iterative symbol-following when no completion endpoint is configured.

### `storage/`: Persistent storage

- `metadata.rs`: SQLite: chunk metadata, file paths, content hashes
- `bm25.rs`: Tantivy: full-text BM25 index
- `vector.rs`: sqlite-vec: embedding vectors

All stored in `.vera/` at the project root.

### `indexing/`: Index build & update

- `pipeline.rs`: Full build: discover → parse → chunk → embed → store
- `update.rs`: Incremental: hash-based change detection, re-process only modified files

### Other modules

- `types.rs`: `Language` enum (60+ variants), `SearchResult`, `CodeChunk`, `SymbolType`
- `config.rs`: `RetrievalConfig`, `IndexConfig` defaults
- `local_models.rs`: Manages local embedding presets, custom ONNX embedding configs, and ORT/model assets under the Vera data directory (XDG-compliant)
- `discovery/`: File discovery with gitignore support, binary/size filtering
- `chunk_text.rs`: Line-boundary text splitting for byte-budget enforcement

## vera-cli

`main.rs` parses args via clap. `commands/` contains the CLI subcommand implementations and helpers: `agent`, `config`, `doctor`, `grep`, `index`, `mcp`, `overview`, `references` (also used by `dead-code`), `repair`, `search`, `setup`, `stats`, `uninstall`, `update`, `upgrade`, and `watch`.

## vera-mcp

`server.rs` routes JSON-RPC requests. `tools.rs` implements four MCP tools: `search_code`, `get_stats`, `get_overview`, and `regex_search`. `search_code` auto-indexes and starts a file watcher on first use.

## Adding a new language

1. Add `Language` variant in `types.rs` (alphabetical order)
2. Add extension mapping in `Language::from_extension()`
3. Add grammar dependency to `vera-core/Cargo.toml`
4. Wire grammar in `parsing/languages.rs` → `tree_sitter_grammar()`
5. Add node classifier in `parsing/extractor.rs` → `classify_node()`
6. Write tests: extension mapping, grammar loading, symbol extraction
