# Vera

Vera is a code indexing and search tool for source trees. It combines lexical search, vector search, and reranking to return ranked code results with file paths, line ranges, symbol metadata, and JSON output that is easy to consume from scripts, editors, and MCP clients.

## Highlights

- Hybrid retrieval: BM25, vector similarity, Reciprocal Rank Fusion, and optional reranking
- Tree-sitter parsing across 60+ languages
- Symbol-aware chunks for functions, methods, classes, structs, and other code units
- Structured JSON output for automation and tool integration
- CLI for direct use and an MCP server for editor and assistant workflows
- API-backed and local inference modes

## Installation

### Prebuilt binaries

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
vera --version
```

### Build from source

Rust 1.85 or newer is required.

```bash
git clone https://github.com/lemon07r/Vera.git
cd Vera
cargo build --release
cp target/release/vera ~/.local/bin/
```

## Configuration

Vera supports two execution modes.

### API mode

Set an embedding endpoint. A reranker is optional but improves result quality.

```bash
export EMBEDDING_MODEL_BASE_URL=https://your-embedding-api/v1
export EMBEDDING_MODEL_ID=your-embedding-model
export EMBEDDING_MODEL_API_KEY=your-api-key

export RERANKER_MODEL_BASE_URL=https://your-reranker-api/v1
export RERANKER_MODEL_ID=your-reranker-model
export RERANKER_MODEL_API_KEY=your-api-key
```

### Local mode

Use `--local` per command or set `VERA_LOCAL=1`. Local models are downloaded on first use to `~/.vera/models/`.

```bash
vera index --local .
vera search --local "authentication logic"

export VERA_LOCAL=1
vera update .
```

## Quick Start

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
vera stats
vera config
```

Vera writes its index to a local `.vera/` directory in the indexed project root.

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

## MCP

Start the MCP server with:

```bash
vera mcp
```

The server exposes:

- `search_code`
- `index_project`
- `update_project`
- `get_stats`

For integration-focused command guidance, see [SKILL.md](SKILL.md).

## Benchmark Snapshot

The benchmark suite in this repository covers 17 tasks across three open-source codebases (`ripgrep`, `flask`, and `fastify`) and five workload categories: symbol lookup, intent search, cross-file discovery, config lookup, and disambiguation.

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
