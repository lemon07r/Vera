# Vera

Code search that actually understands what you're looking for.

Vera is a code indexing and retrieval engine that combines BM25 keyword search, vector similarity, Reciprocal Rank Fusion, and cross-encoder reranking into a single pipeline. It parses 60+ languages with tree-sitter, runs fully local by default, and returns structured, ranked results that work for both humans in the terminal and agents that need machine-readable context.

## Quick Start

```bash
# Install
npx -y @vera-ai/cli install    # or: bunx @vera-ai/cli install / uvx vera-ai install

# Set up local models and index your repo
vera setup
vera index .

# Search
vera search "authentication logic"
vera search "error handling" --lang rust
vera search "handler" --type function --limit 5 --json
```

## Why Vera?

**Hybrid ranking pipeline** — Vera doesn't rely on a single retrieval method. It runs BM25 and vector search in parallel, fuses results with RRF, then applies a cross-encoder reranker. This consistently outperforms any single method alone, especially on ambiguous, intent-heavy queries like `"where does request validation happen"`.

**Fully local by default** — `vera setup` downloads quantized ONNX models to `~/.vera/models/` and indexes your repo to `.vera/`. No hosted service, no API keys, no data leaving your machine. You can optionally point Vera at any OpenAI-compatible endpoint if you prefer.

**Real local models, not a fallback** — The built-in stack pairs a retrieval-tuned embedding model with a dedicated reranker. This preserves the full hybrid ranking pipeline locally, rather than dropping to a single lightweight embedding model like most tools do.

**60+ languages via tree-sitter** — Vera parses source code structurally, extracting functions, classes, methods, and other symbols. Results include file paths, line ranges, content, scores, and symbol metadata.

**Structured output for agents** — JSON output, a CLI workflow agents can call directly, and an installable skill so agents know when and how to use Vera. See [skills/vera/SKILL.md](skills/vera/SKILL.md).

**Benchmark-backed** — On the [public benchmark](#benchmark-snapshot), Vera hybrid reaches `0.6009` MRR@10 and `0.7549` Recall@10 across symbol lookup, intent search, cross-file discovery, config lookup, and disambiguation tasks.

## Installation

### Preferred: CLI + Skills

```bash
npx -y @vera-ai/cli install
bunx @vera-ai/cli install
uvx vera-ai install
```

This installs the `vera` binary for your platform, adds a persistent `vera` command, and runs `vera agent install` for supported agent skill directories.

Then set up and run your first search:

```bash
vera setup
vera index .
vera search "authentication logic"
```

Use `vera doctor` if setup fails.

### Alternative: MCP Server

If your client wants a stdio MCP server instead of CLI + Skills:

```bash
npx -y @vera-ai/cli mcp
bunx @vera-ai/cli mcp
uvx vera-ai mcp
```

Or if `vera` is already installed:

```bash
vera mcp
```

The server exposes `search_code`, `index_project`, `update_project`, and `get_stats`.

### Alternative: Prebuilt Binaries

Releases are published on [GitHub Releases](https://github.com/lemon07r/Vera/releases).

| Platform | Target | Archive |
|----------|--------|---------|
| Linux x86_64 | `x86_64-unknown-linux-gnu` | `.tar.gz` |
| Linux aarch64 | `aarch64-unknown-linux-gnu` | `.tar.gz` |
| macOS x86_64 | `x86_64-apple-darwin` | `.tar.gz` |
| macOS aarch64 | `aarch64-apple-darwin` | `.tar.gz` |
| Windows x86_64 | `x86_64-pc-windows-msvc` | `.zip` |

```bash
curl -sL https://github.com/lemon07r/Vera/releases/latest/download/vera-x86_64-unknown-linux-gnu.tar.gz | tar xz
chmod +x vera-x86_64-unknown-linux-gnu/vera
cp vera-x86_64-unknown-linux-gnu/vera ~/.local/bin/
vera agent install
vera setup
```

### Alternative: Build From Source

Rust 1.85 or newer required.

```bash
git clone https://github.com/lemon07r/Vera.git
cd Vera
cargo build --release
cp target/release/vera ~/.local/bin/
vera agent install
vera setup
```

## Model Backend

Vera itself is always local — the index lives in `.vera/`, config in `~/.vera/`. The backend choice only affects where embeddings and reranking run.

### Built-In Local Models (Default)

`vera setup` downloads quantized ONNX models into `~/.vera/models/` and runs inference locally via ONNX Runtime:

- **Embeddings:** `jinaai/jina-embeddings-v5-text-nano-retrieval`
- **Reranker:** `jinaai/jina-reranker-v2-base-multilingual`

These are modern, retrieval-specific models chosen for quality-per-size. The reranker materially improves ambiguous queries after the initial retrieval pass — this is what separates Vera's local mode from tools that only ship a single embedding model.

### Any OpenAI-Compatible Endpoint

Use `vera setup --api` to point Vera at your own endpoint. This works with remote APIs or local servers like `llama.cpp`.

```bash
export EMBEDDING_MODEL_BASE_URL=https://your-embedding-api/v1
export EMBEDDING_MODEL_ID=your-embedding-model
export EMBEDDING_MODEL_API_KEY=your-api-key

# Optional reranker
export RERANKER_MODEL_BASE_URL=https://your-reranker-api/v1
export RERANKER_MODEL_ID=your-reranker-model
export RERANKER_MODEL_API_KEY=your-api-key

vera setup --api
```

Only model calls leave your machine. Indexing, storage, and search remain local.

## Usage

```bash
vera search "parse_config"
vera search "authentication logic"
vera search "error handling" --lang rust
vera search "routes" --path "src/**/*.ts"
vera search "handler" --type function --limit 5 --json
```

Update after code changes:

```bash
vera update .
```

Inspect the index:

```bash
vera doctor
vera stats
vera config
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

## Benchmark Snapshot

The benchmark suite covers 17 tasks across three open-source codebases (`ripgrep`, `flask`, `fastify`) and five workload categories: symbol lookup, intent search, cross-file discovery, config lookup, and disambiguation. Full details: [docs/benchmarks.md](docs/benchmarks.md).

| Metric | ripgrep | cocoindex-code | vector-only | Vera hybrid |
|--------|---------|----------------|-------------|-------------|
| Recall@5 | 0.2817 | 0.3730 | 0.4921 | **0.6961** |
| Recall@10 | 0.3651 | 0.5040 | 0.6627 | **0.7549** |
| MRR@10 | 0.2625 | 0.3517 | 0.2814 | **0.6009** |
| nDCG@10 | 0.2929 | 0.5206 | 0.7077 | **0.8008** |

- BM25-only search: `3.5 ms` p95 latency
- API-backed hybrid search: `6749 ms` p95 (dominated by remote model calls)
- Indexing `ripgrep` (~175K LOC): `65.1 s`
- Incremental updates: seconds for small changes

More detail: [docs/benchmarks.md](docs/benchmarks.md) · [benchmarks/indexing-performance.md](benchmarks/indexing-performance.md) · [benchmarks/reports/reproduction-guide.md](benchmarks/reports/reproduction-guide.md)

## Supported Languages

Vera supports 60+ languages and file formats, including Python, TypeScript, JavaScript, Java, Go, Rust, C, C++, C#, Ruby, Swift, Kotlin, PHP, Scala, Dart, Haskell, Elixir, Lua, SQL, HTML, CSS, Vue, Terraform, and common config formats like TOML, YAML, JSON, and Markdown.
