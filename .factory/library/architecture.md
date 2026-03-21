# Architecture

Architectural decisions, patterns, and conventions for Vera.

**What belongs here:** Decided architecture patterns, module ownership, key design decisions.
**What does NOT belong here:** Speculative ideas (those go in ADRs until decided).

---

## Decided (updated as ADRs are finalized)

### ADR-001: Implementation Language → Rust
- Rust chosen over TypeScript/Bun based on spike benchmarks
- 1.6–1.8× faster tree-sitter parsing, 10× faster cold start vs Bun
- Single binary distribution (~10-15MB estimated) vs 32MB+ node_modules
- Superior ecosystem: `ignore` crate, Tantivy, Lance, Clap
- See `docs/adr/001-implementation-language.md` for full evidence

### ADR-002: Storage Backend → SQLite + sqlite-vec + Tantivy
- SQLite + sqlite-vec for metadata and vector search, Tantivy for BM25
- Chosen over LanceDB despite LanceDB's 5× faster vector queries and 32× faster writes
- SQLite performance is sufficient: 10ms vector query p50, 7.6K chunks/sec writes
- Key advantage: ~60 crates vs 537 (LanceDB), 40s vs 150s build time, sync API
- Tantivy BM25 is sub-millisecond (0.067ms p50) — uncontested for full-text search
- See `docs/adr/002-storage-backend.md` for full evidence

### ADR-003: Embedding Model → Qwen3-Embedding-8B
- Qwen3-Embedding-8B chosen over bge-en-icl and Qwen3-Embedding-0.6B
- Best Recall@10 (0.66), nDCG (0.71), and Recall@5 (0.49) on Vera's 21-task suite
- Outperforms all M1 competitor baselines on recall and nDCG
- 4096-dim vectors; OpenAI-compatible API via Nebius (EMBEDDING_MODEL_BASE_URL)
- Qwen3-Embedding-0.6B (1024-dim) designated as lightweight fallback for local use
- MRR (0.28) lags cocoindex-code (0.35), confirming reranking is essential
- See `docs/adr/003-embedding-model.md` for full evidence

### ADR-004: Chunking Strategy → Symbol-Aware (tree-sitter AST)
- Symbol-aware chunking chosen over sliding-window and file-level
- 2.3× higher MRR on symbol lookup (0.55 vs 0.24) — correct definitions rank ~2nd vs ~4th
- Best overall MRR (0.38 vs 0.28) and Recall@5 (0.59 vs 0.49)
- 14% more token-efficient than sliding-window (0.86 ratio)
- Intent search R@5=0.90 vs 0.50 for sliding-window
- Sliding-window as Tier 0 fallback for unsupported languages
- See `docs/adr/004-chunking-strategy.md` for full evidence

### ADR-000: Decision Summary
- All 5 major architecture questions decided: language, storage, embedding, chunking, retrieval pipeline shape
- Retrieval pipeline: BM25 (Tantivy) + Vector (sqlite-vec) → RRF fusion → Reranking (Qwen3-Reranker)
- 2 open questions remain: pipeline parameters (tune in M2), graph-lite scope (defer to M2/M3)
- 13 prior assumptions validated, 2 invalidated (LanceDB, sliding-window role), 7 hypothetical
- See `docs/adr/000-decision-summary.md` for full decision matrix and assumption categorization

## Baseline Findings from M1 Competitor Benchmarks

Key insights from competitor baseline benchmarking (21 tasks, 4 repos):

- **Lexical (ripgrep):** Recall@10=0.37, MRR=0.26, p50=18ms. Fast but poor semantic coverage.
- **Semantic (cocoindex-code):** Recall@10=0.50, MRR=0.35, p50=446ms. Balanced quality/speed.
- **Vector-only (Qwen3):** Recall@10=0.66, MRR=0.28, p50=1186ms. Best recall but slow and poor MRR.

**Design implications:**
1. Config lookup tasks are challenging for all tools. Consider filename/filetype matching stage.
2. Cross-file discovery is weak across all baselines (max Recall@10=0.44). Graph-lite metadata may help.
3. Hybrid BM25+vector is clearly justified: lexical is fast for identifiers, vector catches semantics.
4. Reranking is likely the key to MRR improvement over vector-only (high recall, poor MRR suggests ranking issue).

## M2 Core Engine Implementation Details

### Index Storage Convention
- Index artifacts are stored in `.vera/` inside the indexed repository root
- Files: `metadata.db` (SQLite, chunk metadata + file hashes), `vectors.db` (sqlite-vec, embeddings), `bm25/` (Tantivy index directory)
- `.vera/` is gitignored in the project root

### Language Round-Trip Mapping Requirement
- Adding a new `Language` variant requires updating **both** `crates/vera-core/src/types.rs` and `crates/vera-core/src/storage/metadata.rs::parse_language()`
- The forward mapping (`Language::as_str()` / extension detection) is not enough by itself because indexed chunks are read back from SQLite via `row_to_chunk()`
- If `parse_language()` is not updated, persisted chunks deserialize as `Language::Unknown`, which breaks `--lang` filtering and returned result metadata for the new language

### Tree-Sitter Grammar Compatibility Workarounds
- `tree-sitter-hcl` is sourced from git rather than crates.io because the repo needed a newer binding than the published crate provided
- SQL and Protobuf parsers are vendored under `crates/tree-sitter-sql/` and `crates/tree-sitter-proto/`
- `crates/vera-core/build.rs` compiles those parser C files directly to avoid duplicate-symbol / linker conflicts from older tree-sitter crates
- Future language-support work should check for grammar version/linkage conflicts before assuming the crates.io tree-sitter crate can be dropped in unchanged

### Matryoshka Vector Truncation
- Qwen3-Embedding-8B produces 4096-dim vectors natively
- Vera truncates to 1024-dim at index time using Matryoshka truncation (first N dimensions)
- This reduces vector storage by 4× with minimal retrieval quality impact
- Query embeddings are also truncated to match stored dimensions
- **Dimension mismatch footgun:** If old indexes exist with different dimensions (e.g., 4096-dim), vector search will fail silently or return errors. Re-index to fix.

### Chunk Line Encoding
- `line_start` and `line_end` are 1-based (converted from tree-sitter's 0-based row indices: `row + 1`)
- `line_end` is the 1-based line number of the last line of the chunk (inclusive in display, but can be used as exclusive upper bound when 0-indexed)
- When extracting content from source: `source_lines[(line_start-1)..line_end]` gives the correct half-open range

### Tuned Embedding Pipeline Defaults
- `batch_size=128`, `max_concurrent_requests=8`, `timeout_secs=60`, `max_stored_dim=1024`
- Batch size increased from 64→128 for better throughput with fewer API round trips
- Rate limit backoff reduced from 6s→2s base for faster recovery
- `max_concurrent_requests=8` balances throughput vs rate limit avoidance

### Query Embedding Cache
- `CachedEmbeddingProvider` wraps any `EmbeddingProvider` with an in-memory LRU cache
- Caches `query_text → embedding_vector`; first query pays API cost, subsequent identical queries <5ms
- CLI search wraps the provider with 512-entry cache by default
- Multi-text batches (indexing) bypass the cache to avoid memory bloat

### Performance Targets (Clarified)
- **BM25-only p95**: <10ms (local computation, no API dependency)
- **Cached hybrid query p95**: <100ms (query embedding served from cache)
- **First-time hybrid query**: Depends on embedding API latency (typically 500ms–7s for remote APIs)
- **BM25 fallback**: Always available when low latency is required; achieves <5ms p95
- The 500ms p95 target from the original mission brief applies to cached/BM25 queries; first-time hybrid queries with remote API depend on API provider latency

### Method Extraction Asymmetry
- Rust `impl` block methods are extracted as individual chunks (via `extract_impl_methods`)
- Python class methods are extracted individually via `extract_python_class_methods`
- C#, PHP, and Dart class members are extracted individually via `extract_general_class_methods`
- Java, TypeScript/JavaScript, C++, Ruby, Kotlin, Swift, Scala, and most other promoted languages still use the generic `emit symbol then return` path for classes/modules/types
- This means symbol lookup for nested class members is language-dependent: some languages return the method chunk directly, while others return the parent class/module chunk
- Large classes are split via large-symbol splitting, providing partial granularity

### Post-Retrieval Filtering Pattern
- Search filters (`--lang`, `--path`, `--type`, `--limit`) are applied post-retrieval, not within BM25/vector search
- Over-fetch strategy: request 3× + 20 extra results from the retrieval pipeline, then filter
- This keeps the core pipeline simple and makes filters composable

### M2 Benchmark Results
- Hybrid MRR@10: 0.60 (vs BM25-only 0.28, vector-only 0.28)
- Reranked Precision@3: 0.245 (vs unreranked 0.137)
- BM25 p95 latency: 3.5ms (local computation)
- Hybrid p95 latency: 6749ms (dominated by external API round trips)
- 175K LOC indexed in 59.2s, 1.38× size ratio

## M3 Agent Integration Details

### Chunk ID Format Convention
- Chunk IDs follow the format `{file_path}:{index}` (e.g., `src/main.rs:0`, `src/main.rs:1`)
- The `file_path` is repo-relative; `index` is a 0-based counter per file
- This convention is critical for `VectorStore.delete_by_file_prefix` which uses `{file_path}:` as a LIKE prefix
- Changing this format would break incremental indexing's prefix-based deletion

### sqlite-vec vec0 Upsert Limitation
- The vec0 virtual table does NOT support INSERT OR REPLACE or ON CONFLICT
- Workaround: DELETE existing vector first, then INSERT new one
- This is why incremental indexing uses `delete_by_file_prefix` before re-inserting updated chunks
- See `crates/vera-core/src/storage/vector.rs` comments at batch_insert for details

### MCP Server Transport
- MCP server uses stdio transport (JSON-RPC 2.0 over stdin/stdout), NOT TCP/HTTP
- No port is used; the server reads from stdin and writes to stdout
- Launched by MCP clients (Claude Desktop, VS Code) — not by direct HTTP connection
- `vera mcp` starts the server; no `--port` flag exists

### JSON Serialization Conventions
- `Language` enum serializes as **lowercase**: `rust`, `python`, `typescript`, `go`, etc. (via `#[serde(rename_all = "lowercase")]`)
- `SymbolType` enum serializes as **snake_case**: `function`, `class`, `struct`, `type_alias`, etc. (via `#[serde(rename_all = "snake_case")]`)
- `SearchResult` always includes all 8 fields: `file_path`, `line_start`, `line_end`, `content`, `language`, `score`, `symbol_name`, `symbol_type`
- `symbol_name` and `symbol_type` are `null` (not omitted) when no symbol is detected

### Shared Search Service
- Search logic is extracted into `crates/vera-core/src/retrieval/search_service.rs`
- Both CLI (`crates/vera-cli/src/commands/search.rs`) and MCP (`crates/vera-mcp/src/tools.rs`) call this shared service
- The service handles BM25 fallback, embedding provider setup, reranker creation, and filter application
- Note: `execute_search()` creates a new tokio runtime per call — acceptable for CLI, suboptimal for high-throughput MCP use

### Shared Update Path
- Incremental update logic is extracted into `crates/vera-core/src/indexing/update.rs::update_repository`
- Both CLI (`crates/vera-cli/src/commands/update.rs`) and MCP (`crates/vera-mcp/src/tools.rs`) call this shared core function
- Provider/model/dimension validation should live in the shared core path, not only in a CLI wrapper, or MCP/direct callers can diverge from CLI behavior

## Local Inference (v1.1)

### Models
- Embedding: jina-embeddings-v5-text-nano-retrieval (239M params, 768-dim, ONNX format)
  - Task-specific retrieval variant with query/document prefixes
  - HuggingFace repo: jinaai/jina-embeddings-v5-text-nano-retrieval (onnx/ subfolder)
  - Last-token pooling, normalize output
- Reranker: jina-reranker-v2-base-multilingual (278M params, ONNX format)
  - Cross-encoder architecture, query+doc pair → relevance score
  - HuggingFace repo: jinaai/jina-reranker-v2-base-multilingual
  - sigmoid on logits for score

### Design
- All local code gated by `#[cfg(feature = "local")]` Cargo feature
- ONNX Runtime via `ort` crate v2
- HuggingFace `tokenizers` crate for local tokenization
- Models stored in ~/.vera/models/ (current code hardcodes this path; no `VERA_MODEL_DIR` override is implemented yet)
- Local embedding and reranking providers wrap `ort::Session` in `Arc<Mutex<_>>`; preserve serialized session access unless future ONNX Runtime changes make shared sessions safer to use concurrently
- Atomic download: write to .tmp then rename
- Index metadata stores model_name + embedding_dim for mismatch detection
- API mode: 4096-dim (Qwen3-8B), Local mode: 768-dim (Jina v5 nano)
- Indexes are model-specific — cannot mix API and local vectors

### Nebius API Rate Limits (from benchmarks)
- Daily embedding quota exhausted after ~600 files across 3 repos
- Cooldowns of 90-180s don't help — limit is daily, not per-minute

## Key Constraints

- Files under 300 lines (soft), 500 lines (hard - must explain)
- Functions under 40 lines (soft), 80 lines (hard - must explain)
- Explicit module ownership boundaries
- Side effects at boundaries only
- Composition over inheritance
- No magic-heavy patterns without justification
