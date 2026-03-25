# Vera

Vera is a fully local code indexing and retrieval tool for source trees. It keeps each repository index in that repository's own `.vera/` directory, returns ranked code results with file paths, line ranges, symbol metadata, and JSON output, and is designed to work well both from the terminal and from coding agents.

Vera combines BM25 keyword search, vector search, Reciprocal Rank Fusion (RRF), and optional reranking. You can use Vera's built-in local ONNX models, or point it at any OpenAI-compatible endpoint (local or remote). The indexing, storage, and search pipeline stay local on your machine; the only thing that changes is where embeddings and reranking are computed.

## Why Vera?

- Local by design. Vera stores each repo index in `.vera/`, keeps user config under `~/.vera/`, and does not depend on any hosted Vera service.
- Higher-quality built-in local defaults. Vera's built-in local path uses a retrieval-tuned embedding model plus a dedicated reranker, not just a single local embedding model, so local search keeps the same hybrid ranking shape Vera is built around.
- Better than grep for intent-heavy search. Queries like `"authentication logic"` or `"where request validation happens"` work without needing the exact symbol name first. Vera is meant to complement tools like `rg`, not replace them, and comes with SKILL files so your agents know how and when to best use Vera alongside other tools.
- Built for coding agents and direct CLI use. Vera has structured JSON output, an installable skill, and a CLI workflow that agents can call directly.
- Strong ranking quality on the [public benchmark snapshot](#benchmark-snapshot). Vera hybrid reaches `0.6009` MRR@10 and `0.7549` Recall@10 across the public benchmark set; see [docs/benchmarks.md](docs/benchmarks.md) for details and caveats.
- Tree-sitter parsing across 60+ languages, with symbol-aware chunks for functions, methods, classes, structs, and other code units.
- Flexible model backends. Use Vera's built-in local models, or connect to any OpenAI-compatible embedding and reranker endpoints, including self-hosted ones.

## Installation

### Preferred: CLI + Skills

Paste one of these:

```bash
npx -y @vera-ai/cli install
bunx @vera-ai/cli install
uvx vera-ai install
```

That installs the right `vera` binary for your platform, adds a persistent `vera` command, and runs `vera agent install` for supported agent skill directories.

Then bootstrap Vera and run your first search:

```bash
vera setup
vera index .
vera search "authentication logic"
```

Use `vera doctor` if setup fails.

### Alternative: MCP

If your client specifically wants a stdio MCP server instead of CLI + Skills, use one of these:

```bash
npx -y @vera-ai/cli mcp
bunx @vera-ai/cli mcp
uvx vera-ai mcp
```

If you already installed `vera`, the equivalent local command is:

```bash
vera mcp
```

The server exposes:

- `search_code`
- `index_project`
- `update_project`
- `get_stats`

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

Rust 1.85 or newer is required.

```bash
git clone https://github.com/lemon07r/Vera.git
cd Vera
cargo build --release
cp target/release/vera ~/.local/bin/
vera agent install
vera setup
```

## Choose A Model Backend

Vera itself is local. The index always lives in the repo's `.vera/` directory. The choice below only changes where embeddings and reranking are computed.

### Built-In Local Models

`vera setup` is the default path. It downloads the default Jina ONNX models into `~/.vera/models/` and uses ONNX Runtime for local inference.

Vera's built-in local stack currently uses:

- `jinaai/jina-embeddings-v5-text-nano-retrieval` for embeddings
- `jinaai/jina-reranker-v2-base-multilingual` for reranking

Why these models:

- They match Vera's actual retrieval pipeline. Vera is not just "embed everything and cosine-search it" - it combines BM25, vector retrieval, RRF fusion, and reranking, so the local default uses both a retrieval-focused embedding model and a dedicated reranker.
- They are quantized ONNX assets, which makes them practical to cache under `~/.vera/models/` and run locally without a hosted Vera service.
- The reranker materially improves ambiguous and intent-heavy queries after the first retrieval pass. That is usually a stronger local setup than tools that only ship a single lightweight embedding model for their local mode.

```bash
vera setup
vera index .
vera search "authentication logic"
```

What this gives you:

- local repo index on disk in `.vera/`
- local model cache under `~/.vera/models/`
- a self-contained default path for private repos and offline-ish workflows once the models are cached

### Any OpenAI-Compatible Endpoint

Use `vera setup --api` when you already have embedding and reranker endpoints. This can still be a fully local setup if you point Vera at a local OpenAI-compatible server such as `llama.cpp` or any other self-hosted stack.

Set these first:

```bash
export EMBEDDING_MODEL_BASE_URL=https://your-embedding-api/v1
export EMBEDDING_MODEL_ID=your-embedding-model
export EMBEDDING_MODEL_API_KEY=your-api-key
```

Optional reranker:

```bash
export RERANKER_MODEL_BASE_URL=https://your-reranker-api/v1
export RERANKER_MODEL_ID=your-reranker-model
export RERANKER_MODEL_API_KEY=your-api-key
```

Then persist the configuration:

```bash
vera setup --api
```

If you point those variables at a remote service, only the model calls leave your machine. Vera still indexes, stores, and searches the codebase locally.

## Quick Start

If `vera` is already installed, install or refresh the Vera skill:

```bash
vera agent install
vera agent status --scope all
```

Index a repository:

```bash
vera index .
```

Search it:

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

For agent-facing Vera usage guidance, see [skills/vera/SKILL.md](skills/vera/SKILL.md).

Sample JSON search result:

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

The benchmark suite in this repository covers 17 tasks across three open-source codebases (`ripgrep`, `flask`, and `fastify`) and five workload categories: symbol lookup, intent search, cross-file discovery, config lookup, and disambiguation. Full details: [docs/benchmarks.md](docs/benchmarks.md).

| Metric | ripgrep | cocoindex-code | vector-only | Vera hybrid |
|--------|---------|----------------|-------------|-------------|
| Recall@5 | 0.2817 | 0.3730 | 0.4921 | **0.6961** |
| Recall@10 | 0.3651 | 0.5040 | 0.6627 | **0.7549** |
| MRR@10 | 0.2625 | 0.3517 | 0.2814 | **0.6009** |
| nDCG@10 | 0.2929 | 0.5206 | 0.7077 | **0.8008** |

Additional performance notes from the same benchmark set:

- `vera search` in BM25-only mode measured `3.5 ms` p95 latency
- API-backed hybrid search measured `6749 ms` p95 latency and is dominated by remote model calls
- Indexing `ripgrep` (about 175K LOC) completed in `65.1 s`
- Incremental updates complete in a few seconds for small changes

More detail:

- Public benchmark summary: [docs/benchmarks.md](docs/benchmarks.md)
- Indexing performance note: [benchmarks/indexing-performance.md](benchmarks/indexing-performance.md)
- Reproduction guide: [benchmarks/reports/reproduction-guide.md](benchmarks/reports/reproduction-guide.md)

## Supported Languages

Vera supports 60+ languages and file formats, including Rust, Python, TypeScript, JavaScript, Go, Java, C, C++, SQL, Terraform, Protobuf, HTML, CSS, Vue, Dockerfile, Astro, TOML, YAML, JSON, and Markdown.
