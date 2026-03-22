# Vera — Code Search for AI Agents

Vera (Vector Enhanced Relevance Agent) is a code indexing and retrieval CLI tool built for AI coding agents. It combines BM25 keyword search with vector similarity search via Reciprocal Rank Fusion (RRF) and cross-encoder reranking, delivering ranked, structured code results across 60+ languages.

## Key Features

- **Hybrid retrieval** — BM25 + vector search + RRF fusion + cross-encoder reranking
- **60+ languages** — Tree-sitter parsing for Rust, Python, TypeScript, JavaScript, Go, Java, C, C++, Ruby, Bash, Kotlin, Swift, Zig, Lua, Scala, C#, PHP, Haskell, Elixir, Dart, SQL, HCL/Terraform, Protobuf, HTML, CSS, SCSS, Vue, GraphQL, CMake, Dockerfile, XML, Objective-C, Perl, Julia, Nix, OCaml, Groovy, Clojure, Common Lisp, Erlang, F#, Fortran, PowerShell, R, MATLAB, D, Fish, Zsh, Luau, Scheme, Racket, Elm, GLSL, HLSL, Svelte, Astro, Makefile, INI, Nginx, Prisma, plus TOML, YAML, JSON, and Markdown
- **Symbol-aware chunking** — Functions, classes, structs become individual search units
- **Structured output** — JSON context capsules with file paths, line ranges, and code content
- **Fast** — Sub-5ms BM25 queries, incremental updates under 5 seconds
- **Local inference** — Optional local ONNX models (Jina v5 embedding + v2 reranker) for offline/private use — no API keys required
- **Single binary** — Rust-native, zero runtime dependencies

## Installation

### Prerequisites

- **Rust 1.85+** — Install via [rustup](https://rustup.rs/)
- **Embedding API** — An OpenAI-compatible embedding endpoint (e.g., Qwen3-Embedding-8B)
- **Reranker API** (optional) — An OpenAI-compatible reranker endpoint for improved precision

### Download Pre-Built Binaries

Pre-built binaries are available from [GitHub Releases](https://github.com/lemon07r/Vera/releases) for:

| Platform | Target | Archive |
|----------|--------|---------|
| Linux x86_64 | `x86_64-unknown-linux-gnu` | `.tar.gz` |
| Linux aarch64 | `aarch64-unknown-linux-gnu` | `.tar.gz` |
| macOS x86_64 (Intel) | `x86_64-apple-darwin` | `.tar.gz` |
| macOS aarch64 (Apple Silicon) | `aarch64-apple-darwin` | `.tar.gz` |
| Windows x86_64 | `x86_64-pc-windows-msvc` | `.zip` |

```bash
# Example: download latest release for Linux x86_64
curl -sL https://github.com/lemon07r/Vera/releases/latest/download/vera-x86_64-unknown-linux-gnu.tar.gz | tar xz
chmod +x vera-x86_64-unknown-linux-gnu/vera
cp vera-x86_64-unknown-linux-gnu/vera ~/.local/bin/

# Verify
vera --version
```

Each release includes SHA256 checksums (`checksums-sha256.txt`) for verification.

### Build from Source

```bash
# Clone and build
git clone https://github.com/lemon07r/Vera.git
cd Vera
cargo build --release

# The binary is at target/release/vera
# Optionally, copy it to your PATH:
cp target/release/vera ~/.local/bin/
```

The single build includes both API and local inference support — no feature flags needed.

### Configure

**Option A: API mode (default)** — requires an OpenAI-compatible embedding endpoint:

```bash
export EMBEDDING_MODEL_BASE_URL=https://your-embedding-api/v1
export EMBEDDING_MODEL_ID=Qwen/Qwen3-Embedding-8B
export EMBEDDING_MODEL_API_KEY=your-api-key

# Optional: enables cross-encoder reranking for better precision
export RERANKER_MODEL_BASE_URL=https://your-reranker-api/v1
export RERANKER_MODEL_ID=Qwen/Qwen3-Reranker
export RERANKER_MODEL_API_KEY=your-api-key
```

**Option B: Local mode** — no API keys needed:

```bash
# Use --local flag or set VERA_LOCAL=1
vera index --local .
vera search --local "authentication logic"

# Or set globally:
export VERA_LOCAL=1
vera index .
vera search "authentication logic"
```

Local mode uses quantized Jina models (~500MB total, downloaded automatically on first use to `~/.vera/models/`):
- **Embedding**: jina-embeddings-v5-text-nano-retrieval (239M params, 768-dim)
- **Reranking**: jina-reranker-v2-base-multilingual (278M params)

### Verify Installation

```bash
vera --version
# vera 0.1.0

vera --help
# Shows available commands: index, search, update, stats, config, mcp
```

## Quick Start

### 1. Index a Project

```bash
vera index .
```

Output:
```
Indexed 209 files (5377 chunks, 5377 embeddings) in 59.2s
```

This creates a `.vera/` directory containing the search index.

### 2. Search for Code

```bash
# Symbol lookup — find function definitions
vera search "parse_config"

# Semantic search — find code by intent
vera search "authentication logic"

# Filter by language
vera search "error handling" --lang rust

# Filter by path
vera search "routes" --path "src/**/*.ts"

# JSON output for programmatic use
vera search "handler" --limit 5 --json
```

Sample JSON output:
```json
[
  {
    "file_path": "src/auth/login.rs",
    "line_start": 42,
    "line_end": 68,
    "content": "pub fn authenticate(credentials: &Credentials) -> Result<Token> { ... }",
    "language": "rust",
    "score": 0.847,
    "symbol_name": "authenticate",
    "symbol_type": "function"
  }
]
```

### 3. Update After Changes

```bash
vera update .
# Updated 1 file (3 chunks added, 2 removed) in 3.1s
```

Only re-indexes files that changed since the last index, detected via content hashing.

### 4. View Statistics

```bash
vera stats
```

Output:
```
Index Statistics
  Files:    209
  Chunks:   5,377
  Size:     32.4 MB
  Languages: rust (143), toml (28), markdown (21), yaml (8), json (5), bash (4)
```

### MCP Server

For tool-use integration with AI agents:

```bash
vera mcp
```

Exposes tools over JSON-RPC stdio: `search_code`, `index_project`, `update_project`, `get_stats`.

For local inference mode via MCP:
```bash
VERA_LOCAL=1 vera mcp
```

## Benchmark Results

Benchmarked on 17 tasks across 3 repositories (ripgrep/Rust, flask/Python, fastify/TypeScript) covering 5 workload categories: symbol lookup, intent search, cross-file discovery, config lookup, and disambiguation.

### Retrieval Quality

| Metric       | ripgrep | cocoindex-code | vector-only | **Vera (hybrid)** |
|--------------|---------|----------------|-------------|-------------------|
| **Recall@1** | 0.15    | 0.16           | 0.10        | **0.46**          |
| **Recall@5** | 0.35    | 0.37           | 0.49        | **0.67**          |
| **Recall@10**| 0.37    | 0.50           | 0.66        | **0.87**          |
| **MRR@10**   | 0.32    | 0.35           | 0.28        | **0.69**          |
| **nDCG@10**  | 0.29    | 0.52           | 0.71        | **0.99**          |

Vera's hybrid pipeline achieves **2.9× higher Recall@1**, **+81% Recall@5**, and **+97% MRR** compared to the best competitor on each metric.

### Per-Category Highlights

| Category          | Best Competitor | Vera Hybrid | Improvement |
|-------------------|-----------------|-------------|-------------|
| Symbol Lookup MRR | 0.34 (cocoindex)| **0.83**    | +144%       |
| Intent Search MRR | 0.63 (cocoindex)| **0.54**    | —           |
| Config Lookup R@5 | 0.75 (vector)   | **1.00**    | +33%        |
| Disambiguation MRR| 0.39 (ripgrep)  | **1.00**    | +156%       |
| Cross-file MRR    | 0.56 (cocoindex)| **0.50**    | —           |

### Performance

| Metric                     | Target    | Actual           |
|----------------------------|-----------|------------------|
| 100K+ LOC indexing         | < 120s    | **65.2s** (175K LOC) |
| BM25 query p95 latency     | < 10ms    | **3.8ms**        |
| Incremental update         | < 5s      | **3.3s**         |
| Index size / source size   | < 2.0×    | **1.64×**        |

### Ablation Summary

| Component              | MRR Impact   | Recommendation |
|------------------------|-------------|----------------|
| + BM25 fusion          | +145% vs vector-only | Essential |
| + Vector search        | +84% vs BM25-only    | Essential |
| + Cross-encoder rerank | +67% vs unreranked    | Recommended |

Each pipeline component addresses different failure modes: BM25 handles exact identifiers, vectors enable semantic search, and reranking improves precision. Full details in [`benchmarks/reports/`](benchmarks/reports/).

## Architecture

Vera is built on evidence-backed architecture decisions documented in [docs/adr/](docs/adr/):

- **Language:** Rust — 1.6–1.8× faster parsing, 10× faster cold start, single binary ([ADR-001](docs/adr/001-implementation-language.md))
- **Storage:** SQLite + sqlite-vec + Tantivy — sufficient performance, 60 vs 537 crate dependencies ([ADR-002](docs/adr/002-storage-backend.md))
- **Embedding:** Qwen3-Embedding-8B — highest Recall@10 (0.66) and nDCG (0.71) on code tasks ([ADR-003](docs/adr/003-embedding-model.md))
- **Chunking:** Symbol-aware via tree-sitter AST — 2.3× higher MRR on symbol lookup ([ADR-004](docs/adr/004-chunking-strategy.md))

## Project Structure

```
crates/
  vera-core/    # Parsing, indexing, retrieval, storage, embedding
  vera-cli/     # CLI interface (clap)
  vera-mcp/     # MCP server (JSON-RPC stdio)
eval/           # Evaluation harness, corpus, benchmark tasks
benchmarks/     # Benchmark results and reports
docs/adr/       # Architecture Decision Records
```

## License

MIT
