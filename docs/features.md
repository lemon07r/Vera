# Features

Vera (Vector Enhanced Reranking Agent) is a code search tool that combines BM25 keyword matching, vector similarity, and cross-encoder reranking into a single retrieval pipeline. This page covers everything it can do.

## Search Pipeline

### Hybrid Retrieval (BM25 + Vector)

Every query runs two retrieval paths in parallel:

- **BM25 keyword search** via Tantivy. Handles exact identifiers, config keys, and literal strings. Sub-millisecond latency.
- **Vector similarity search** via sqlite-vec. Catches conceptual matches even when the exact words don't appear in the code.

Results from both paths merge through Reciprocal Rank Fusion (RRF), so a result that scores well in both lists rises to the top. Full details: [how-it-works.md](how-it-works.md).

### Cross-Encoder Reranking

After fusion, the top candidates go through a cross-encoder that reads query and candidate together as a single pair. This is the most impactful stage: it lifts MRR@10 from 0.39 to 0.60 (54% improvement). Most code search tools skip this step entirely.

### Multi-Query Search

A single search call can accept multiple queries at once. Run 2-3 varied queries to capture different aspects of what you're looking for (e.g., "OAuth token refresh", "JWT expiry handling", "auth middleware"). Results are deduplicated and reranked together. This reduces round-trips and improves recall.

### Intent-Based Reranking

An optional `intent` parameter lets you describe your higher-level goal separately from the search query. The reranker uses this to score candidates against what you actually need, not just what you typed. Useful when the query is ambiguous or too short to convey full context.

### Multi-Hop Deep Search

`vera search "query" --deep` runs an initial search, extracts symbol names from the top results, then automatically searches for those symbols to find related code. This follows the call chain outward from your initial results without manual follow-up queries.

### Query-Aware Ranking

A deterministic ranking stage between fusion and reranking handles cases that dense retrieval alone is bad at: exact filename queries, path-heavy config lookups, noisy test/docs matches, and broad natural-language queries that need structural results. It also pulls in related implementation blocks or same-file context when the initial hit is too narrow.

## Parsing and Indexing

### Tree-Sitter Structural Parsing

63 languages parsed with tree-sitter grammars compiled into the binary. Functions, classes, structs, traits, interfaces, methods, and `impl` blocks are extracted as discrete chunks. Results map to actual symbol boundaries, not arbitrary line ranges.

Symbol-aware chunking scores 2.3x higher MRR on symbol lookup than sliding-window chunking (0.55 vs 0.24), while using 14% fewer tokens. Large symbols (>150 lines) are split at logical boundaries. Full list: [supported-languages.md](supported-languages.md).

### Incremental Updates

`vera update .` compares content hashes against the stored index and only re-parses, re-chunks, and re-embeds changed files. For small changes this takes seconds, not minutes.

### File Watching

`vera watch .` monitors the project for file changes and triggers incremental index updates automatically (debounced at 2s). Keeps the index fresh during long coding sessions without manual intervention.

### Flexible Exclusions

Vera respects `.gitignore` by default. For more control, `.veraignore` (gitignore syntax) gives full control over what gets indexed. Use `#include .gitignore` at the top to layer extra exclusions on top of gitignore rules. One-off `--exclude` flags work too.

## Code Intelligence

### Call Graph and Reference Finding

`vera references foo` finds all callers of a symbol. `vera references foo --callees` finds what a symbol calls. The call graph is built during indexing from tree-sitter AST analysis, so lookups are instant.

### Dead Code Detection

`vera dead-code` finds functions and methods with no callers. Excludes common entry points (`main`, `new`, `default`, etc.) to reduce noise. Useful for codebase cleanup and understanding which code is actually reachable.

### Project Overview

`vera overview` generates an architecture summary: language breakdown, directory structure, entry points, symbol type distribution, complexity hotspots, and detected project conventions (frameworks, patterns, config files). Designed for agent onboarding so an AI agent can understand a codebase before diving in.

### Regex Search

`vera grep "pattern"` runs regex search over all indexed files with configurable context lines and case sensitivity. Complements semantic search for exact string matching, import statements, TODOs, and known identifiers.

## Model Backend

### Local-First, Model-Agnostic

Indexing, storage, and search always stay on your machine. The backend choice only affects where embeddings and reranking run:

- **Local mode**: `vera setup` downloads curated ONNX models. The full pipeline (BM25 + vector + rerank) runs without external calls.
- **API mode**: Point at any OpenAI-compatible endpoint (remote APIs or local servers like llama.cpp). Only model calls leave your machine.

### Curated Local Models

Two quantized ONNX models ship with local mode:

| Model | Role |
|-------|------|
| [jina-embeddings-v5-text-nano-retrieval](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval) | Default embedding model |
| [jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | Cross-encoder reranker |

An optional [CodeRankEmbed](https://huggingface.co/Zenabius/CodeRankEmbed-onnx) preset is available for embedding-heavy or no-rerank experiments. Details: [models.md](models.md).

### GPU Acceleration

Auto-detected during setup. Supported backends:

| Flag | Hardware |
|------|----------|
| `--onnx-jina-cuda` | NVIDIA (CUDA 12+) |
| `--onnx-jina-rocm` | AMD (Linux, ROCm) |
| `--onnx-jina-directml` | Any DirectX 12 GPU (Windows) |
| `--onnx-jina-coreml` | Apple Silicon (macOS) |
| `--onnx-jina-openvino` | Intel GPU/iGPU (Linux) |

Vera downloads the matching ONNX Runtime build automatically. For OpenVINO and ROCm, it installs via pip into a managed venv.

### Adaptive GPU Batching

Local ONNX indexing shapes micro-batches from actual token lengths rather than using a fixed batch size. Long inputs shrink roughly with the square of padded sequence length. Vera learns safe batch windows per length bucket from real successes and allocation failures, persists them across runs, and warm-starts on the next session. On constrained GPUs, pass `--low-vram` to force conservative settings.

### Custom Local Embeddings

Swap the local embedding model without changing the rest of the pipeline. Point at a Hugging Face repo, a direct URL, or a local directory with custom pooling, query prefix, and dimension settings. The local reranker stays on the curated Jina model.

## Output and Integration

### Token-Efficient Output

Default markdown codeblock format cuts ~35-40% tokens vs JSON. On a 20-query benchmark, Vera's chunk-level output averages 67% fewer tokens than loading the full files containing the same results. Most queries see 75-95% reduction.

### Response Truncation

Large chunks are automatically truncated at 8K characters with a `[...truncated]` marker to prevent blowing up LLM context windows. Short results pass through unchanged.

### Multiple Output Formats

| Flag | Output |
|------|--------|
| *(default)* | Markdown codeblocks with file path, line range, and symbol metadata |
| `--json` | Compact single-line JSON |
| `--raw` | Verbose human-readable output |
| `--timing` | Per-stage pipeline durations to stderr |

### Search Filters

Narrow results by language (`--lang rust`), file path glob (`--path "src/**/*.rs"`), symbol type (`--type function`), corpus scope (`--scope source`), and result count (`--limit 5`). Filters combine, so `--lang rust --type function --path "src/**"` returns only Rust functions under `src/`.

## MCP Server

`vera mcp` exposes all capabilities over JSON-RPC (stdio), compatible with any MCP client:

| Tool | What it does |
|------|-------------|
| `search_code` | Hybrid search with multi-query, intent, and all filters |
| `index_project` | Full index build |
| `update_project` | Incremental update |
| `get_stats` | File count, chunk count, index size, language breakdown |
| `get_overview` | Architecture overview with conventions detection |
| `watch_project` | Auto-update index on file changes |
| `find_references` | Callers or callees of a symbol |
| `find_dead_code` | Functions with no callers |
| `regex_search` | Regex search with context lines |

Tool descriptions include explicit WHEN TO USE / WHEN NOT TO USE guidance so AI agents route queries to the right tool automatically.

Docker images available for CPU, CUDA, ROCm, and OpenVINO. Details: [docker.md](docker.md).

## Agent Integration

### Skill Files

`vera agent install` installs skill files that teach AI agents how to write effective queries, when to use semantic search vs regex, and how to interpret results. Supports Junie, Claude Code, Cursor, Windsurf, Copilot, Cline, and Roo Code. Skills install globally or per-project.

### Agent Config Snippets

During setup, Vera offers to add a usage snippet to your project's agent config file (`AGENTS.md`, `CLAUDE.md`, `.cursorrules`, etc.) so agents discover Vera automatically.

### Auto-Sync on Upgrade

`vera upgrade --apply` automatically syncs stale agent skill installs to match the new binary version after upgrading.

## CLI Tooling

### Interactive Setup Wizard

`vera setup` walks through backend selection, agent skill installation, and optional project indexing in one command. Skip the wizard with flags for non-interactive use.

### Diagnostics

`vera doctor` reports the saved and active backend, installed version, and checks GitHub for newer releases. `--probe` adds a deeper read-only ONNX session check. `vera repair` re-fetches missing local assets.

### Self-Updating

`vera upgrade` inspects the current update plan and shows the exact command it would run. `--apply` executes it. Vera checks for new releases once per day and prints a hint when a newer version is available.

### Cross-Platform

Single static binary for Linux (x86_64, aarch64), macOS (x86_64, aarch64), and Windows (x86_64). Install via npm (`bunx @vera-ai/cli install`), pip (`uvx vera-ai install`), prebuilt binary, Docker, or build from source.

## Benchmarks

On a 21-task benchmark across `ripgrep`, `flask`, `fastify`, and `turborepo`:

| Metric | v0.4.0 | v0.7.0+ |
|--------|--------|---------|
| Recall@1 | 0.24 | **0.72** |
| Recall@5 | 0.50 | **0.78** |
| MRR@10 | 0.50 | **0.91** |
| nDCG@10 | 0.46 | **0.84** |

Comparison against other tools on the same workload: [benchmarks.md](benchmarks.md).
