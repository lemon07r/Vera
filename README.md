# Vera

**V**ector **E**nhanced **R**eranking **A**gent

Vera is a code search tool built in Rust that combines BM25 keyword matching, vector similarity, and cross-encoder reranking into a single retrieval pipeline. It parses 60+ languages with tree-sitter, runs everything locally, and returns structured JSON with file paths, line ranges, symbol metadata, and relevance scores.

## Quick Start

```bash
# Install
npx -y @vera-ai/cli install    # or: bunx @vera-ai/cli install / uvx vera-ai install

# Option A: use any OpenAI-compatible endpoint
export EMBEDDING_MODEL_BASE_URL=https://your-api/v1
export EMBEDDING_MODEL_ID=your-model
export EMBEDDING_MODEL_API_KEY=your-key
vera setup --api

# Option B: download Vera's curated local models (no API needed)
vera setup

# Index and search
vera index .
vera search "authentication logic"
vera search "error handling" --lang rust
vera search "handler" --type function --limit 5 --json
```

## Why Vera?

**The reranking stage is what matters.** Most code search tools stop at embeddings. They encode your query and your code into vectors, compare them, and return the closest matches. This works for exact symbol lookups but falls apart on intent queries like `"where does request validation happen"` because embedding similarity treats the query and document independently. Vera adds a cross-encoder reranker as a third stage: it reads the query and each candidate chunk together as a pair and scores them jointly. On Vera's benchmark suite, this is the difference between 0.28 MRR@10 (vector-only) and 0.60 MRR@10 (full pipeline). That 2x jump in ranking quality comes almost entirely from the reranker.

**Concrete benchmark numbers.** Vera's benchmark covers 17 tasks across three real codebases (ripgrep, flask, fastify) spanning symbol lookup, intent search, cross-file discovery, config lookup, and disambiguation. Against ripgrep (grep, not a search tool) and cocoindex-code (embedding-only), Vera's hybrid pipeline scores 0.80 nDCG@10 and 0.75 Recall@10. These aren't synthetic benchmarks on curated datasets; they test the kinds of queries developers and agents actually make. Full methodology and reproduction steps are public: [docs/benchmarks.md](docs/benchmarks.md).

**Your model, local or remote.** Vera's pipeline is model-agnostic. Point it at any OpenAI-compatible embedding or reranker endpoint, whether that's a remote API or a local server like llama.cpp. Everything else (indexing, storage, search logic) stays on your machine regardless. If you don't want to manage models at all, `vera setup` downloads two curated ONNX models that run locally via ONNX Runtime, giving you the full three-stage pipeline without any network calls. Most local-first search tools only ship an embedding model and skip reranking entirely, which caps their precision on anything beyond exact name matches. Details on the bundled models are in the [Model Backend](#model-backend) section below.

## Features

**Tree-sitter structural parsing.** Vera builds its index using tree-sitter grammars for 60+ languages, extracting functions, classes, methods, structs, and other symbols as discrete chunks rather than splitting files by line count. This means you can request `--type function --json` and get back precisely the function definitions matching your query, not a list of matching lines with no context boundaries.

**Structured JSON output.** Every search result includes the file path, exact line range, full source content, symbol name, symbol type, language, and relevance score. Agents and scripts can consume this directly without parsing or guessing at context boundaries. A CLI skill file is included so agents know when and how to invoke Vera: [skills/vera/SKILL.md](skills/vera/SKILL.md).

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

Vera itself is always local: the index lives in `.vera/`, config in `~/.vera/`. The backend choice only affects where embeddings and reranking run.

### Curated Local Models

`vera setup` downloads quantized ONNX models into `~/.vera/models/` and runs inference locally via ONNX Runtime:

- **Embeddings:** [`jinaai/jina-embeddings-v5-text-nano-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval) (quantized). A retrieval-focused embedding model from the Jina v5 family, designed for asymmetric search where the query is short and the document is a code block. The nano variant keeps memory usage low while retaining strong retrieval accuracy on technical content.
- **Reranker:** [`jinaai/jina-reranker-v2-base-multilingual`](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) (quantized). A cross-encoder that scores query-document pairs directly rather than comparing pre-computed embeddings. This is the component that makes the biggest difference on ambiguous queries: it sees the full context of both the query and the candidate chunk, catching semantic matches that vector similarity alone would rank lower.

Having both models locally means the full three-stage pipeline (BM25, vector search, rerank) runs without any external calls.

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
