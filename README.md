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

# Option B: download curated local models (no API needed)
vera setup

# Index and search
vera index .
vera search "authentication logic"
vera search "error handling" --lang rust
vera search "handler" --type function --limit 5 --json
```

## Why Vera?

### Cross-encoder reranking

Most code indexing tools retrieve candidates and stop there. Vera adds a cross-encoder reranking stage that reads query and candidate together as a single pair, scoring relevance jointly instead of comparing pre-computed vectors. This is the difference between 0.28 MRR@10 (vector retrieval alone) and 0.60 MRR@10 (with reranking).

### Built for real code questions

`rg` is still the right tool for exact strings and regex patterns. Vera is for intent queries like `"authentication logic"` or `"where does request validation happen"`, where you want ranked, cross-file, symbol-aware results instead of raw line matches.

### Benchmarked against real workloads

17 tasks across three real codebases (ripgrep, flask, fastify). Vera's hybrid pipeline scores 0.80 nDCG@10 and 0.75 Recall@10 against grep-based and embedding-only baselines. Full methodology: [docs/benchmarks.md](docs/benchmarks.md).

## Features

### Tree-sitter structural parsing

Vera uses tree-sitter grammars for 60+ languages to extract functions, classes, methods, and structs as discrete chunks. Search results map to actual symbol boundaries, not arbitrary line ranges. Filter by type with `--type function` or `--type class` to narrow results to exactly the kind of symbol you need.

### Model-agnostic, local-first

Point Vera at any OpenAI-compatible embedding or reranker endpoint, remote or local. Everything else (indexing, storage, search logic) stays on your machine regardless, no cloud hosted services needed. Or run `vera setup` to download two curated ONNX models and run the full pipeline offline. Details: [Model Backend](#model-backend).

### Structured, code-aware results

Every result includes file path, line range, source content, symbol name, symbol type, language, and relevance score. Agents and scripts consume this directly without parsing. A CLI skill file is included for agent integration: [skills/vera/SKILL.md](skills/vera/SKILL.md).

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

- **Embeddings:** [`jinaai/jina-embeddings-v5-text-nano-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval) (quantized ONNX). At 239M parameters, this is the highest scoring embedding model under 500M on MMTEB (65.5), beating KaLM-mini-v2.5 (494M), Gemma-300M (308M), and voyage-4-nano (480M). It scores 71.0 on MTEB English v2. Built on EuroBERT-210M with distillation from Qwen3-Embedding-4B, it uses a retrieval-specific LoRA adapter designed for asymmetric search where the query is short and the document is a code block.
- **Reranker:** [`jinaai/jina-reranker-v2-base-multilingual`](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) (quantized ONNX). A 278M parameter cross-encoder that scores query-document pairs jointly rather than comparing pre-computed embeddings. Its 1,024-token context window is a natural fit for Vera's tree-sitter symbol chunks: discrete functions and classes, not raw files. Fine-tuned on ToolBench (function-calling schemas) and NSText2SQL (structured queries), it scores 71.36 on CodeSearchNet MRR@10 and 77.75 on ToolBench recall@3 while being half the size and 15x faster than bge-reranker-v2-m3 (568M).

With both models cached locally, the full three-stage pipeline (BM25, vector search, rerank) runs without any external calls. This gives you:

- A local repo index on disk in `.vera/`
- A local model cache under `~/.vera/models/`
- A fully self-contained setup for private repos and offline workflows

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

Install agent skill files and check status:

```bash
vera agent install
vera agent status --scope all
```

Inspect the index:

```bash
vera doctor
vera stats
vera config
```

The skill file at [skills/vera/SKILL.md](skills/vera/SKILL.md) teaches agents how to use Vera effectively.

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
