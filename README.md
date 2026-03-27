<img width="1584" height="539" alt="vera" src="https://github.com/user-attachments/assets/c866fc70-b1e6-400b-aaf7-fa68721a4955" />

# Vera

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/lemon07r/Vera/blob/master/Cargo.toml)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![GitHub release](https://img.shields.io/github/v/release/lemon07r/Vera?include_prereleases&sort=semver)](https://github.com/lemon07r/Vera/releases)
[![Languages](https://img.shields.io/badge/languages-63%2B-green.svg)](docs/supported-languages.md)
[![npm](https://img.shields.io/npm/v/@vera-ai/cli)](https://www.npmjs.com/package/@vera-ai/cli)
[![PyPI](https://img.shields.io/pypi/v/vera-ai)](https://pypi.org/project/vera-ai/)

**V**ector **E**nhanced **R**eranking **A**gent

Vera is a code search tool built in Rust that combines BM25 keyword matching, vector similarity, and cross-encoder reranking into a single retrieval pipeline. It parses 60+ languages with tree-sitter, runs everything locally, and returns structured JSON with file paths, line ranges, symbol metadata, and relevance scores.

After trying many other tools and maintaining Pampax, a fork of someone's code search tool, I ran into constant issues. Pampax was built on vibeslop code with deep-rooted bugs, and no matter how much I fixed and improved it, it stayed slow and fragile. Nothing out there supported all the things I wanted (like provider-agnostic reranking), so I set out to build something better from scratch. Every design choice in Vera (the retrieval pipeline, the model selection, the output format) comes from hours of research, real benchmarking and evaluation, not guesswork.

## Table of Contents

- [Quick Start](#quick-start)
- [Why Vera is Better](#why-vera-is-better)
- [Features](#features)
- [Installation](#installation)
- [Model Backend](#model-backend)
- [Usage](#usage)
- [Benchmark Snapshot](#benchmark-snapshot)
- [Supported Languages](#supported-languages)
- [How It Works](#how-it-works)
- [Contributing](#contributing)

**Docs:** [Query Guide](docs/query-guide.md) · [Benchmarks](docs/benchmarks.md) · [How It Works](docs/how-it-works.md) · [Models](docs/models.md) · [Docker](docs/docker.md) · [Supported Languages](docs/supported-languages.md) · [Troubleshooting](docs/troubleshooting.md)

## Quick Start

```bash
bunx @vera-ai/cli install   # or: npx -y @vera-ai/cli install / uvx vera-ai install
vera setup                   # downloads local models, no API needed
vera index .
vera search "authentication logic"
```

## Why Vera is Better

### Cross-encoder reranking

Most code indexing tools retrieve candidates and stop there. Vera adds a cross-encoder reranking stage that reads query and candidate together as a single pair, scoring relevance jointly instead of comparing pre-computed vectors. This is the difference between 0.28 MRR@10 (vector retrieval alone) and 0.60 MRR@10 (with reranking).

### Zero-dependency, single binary

Vera ships as one static binary with all 60+ language grammars compiled in via tree-sitter. No Python runtime, no language servers, no per-language toolchains to install or manage. Drop the binary on any machine, run `vera setup`, and the full search pipeline is ready. Tools like Serena require a Python runtime and uv just to start, plus separate LSP dependencies for some languages. Vera has zero external dependencies.

### Higher accuracy, proven on real codebases

On a 17-task benchmark across `ripgrep`, `flask`, and `fastify`, Vera's hybrid pipeline scores `0.80` nDCG@10 and `0.70` Recall@5, compared to `0.52` nDCG@10 for cocoindex-code and `0.71` for vector-only search. The current 21-task suite scores even higher. See [Benchmark Snapshot](#benchmark-snapshot) for the full numbers.

### Token-efficient output, built for AI agents

Vera defaults to markdown codeblocks, cutting output size ~35-40% compared to typical JSON. It ships with skill files that teach agents how to write effective queries, what filters to use, and when to reach for `rg` instead.

## Features

### Model-agnostic, local-first

Point Vera at any OpenAI-compatible embedding or reranker endpoint, remote or local. Everything else (indexing, storage, search logic) stays on your machine regardless, no cloud hosted services needed. Run `vera setup` to download two curated ONNX models and run the full pipeline offline. Details: [Model Backend](#model-backend).

### Tree-sitter structural parsing

Vera uses tree-sitter grammars for 60+ languages to extract functions, classes, methods, and structs as discrete chunks. Search results map to actual symbol boundaries, not arbitrary line ranges. Filter by type with `--type function` or `--type class` to narrow results to exactly the kind of symbol you need.


### Structured, code-aware results

Every result includes file path, line range, source content, symbol name, and symbol type. Agents and scripts consume this directly without parsing. See [AGENT-USAGE.md](AGENT-USAGE.md) for AI agent integration.

## Installation

```bash
bunx @vera-ai/cli install   # or: npx -y @vera-ai/cli install / uvx vera-ai install
```

This downloads the `vera` binary, adds it to your PATH, and installs agent skill files. After this, `vera` is a standalone command. You don't need `bunx`/`npx`/`uvx` again.

```bash
vera setup          # download local models (or vera setup --api for remote endpoints)
vera index .        # index the current project (creates .vera/ in project root)
vera search "query" # search (each project gets its own index)
vera update .       # after code changes
```

Use `vera doctor` if anything goes wrong.

<details>
<summary>MCP server (JSON-RPC over stdio)</summary>

```bash
vera mcp   # or: bunx @vera-ai/cli mcp / uvx vera-ai mcp
```

Exposes `search_code`, `index_project`, `update_project`, and `get_stats` tools.

</details>

<details>
<summary>Docker (MCP server)</summary>

```bash
docker run --rm -i -v $(pwd):/workspace ghcr.io/lemon07r/vera:cpu
```

CPU, CUDA (NVIDIA), ROCm (AMD), and OpenVINO (Intel) images available. See [docs/docker.md](docs/docker.md) for GPU flags and MCP client configuration.

</details>

<details>
<summary>Prebuilt binaries</summary>

Download from [GitHub Releases](https://github.com/lemon07r/Vera/releases) for Linux (x86_64, aarch64), macOS (x86_64, aarch64), or Windows (x86_64).

```bash
curl -sL https://github.com/lemon07r/Vera/releases/latest/download/vera-x86_64-unknown-linux-gnu.tar.gz | tar xz
cp vera-x86_64-unknown-linux-gnu/vera ~/.local/bin/
vera agent install && vera setup
```

</details>

<details>
<summary>Build from source</summary>

Rust 1.85+ required.

```bash
git clone https://github.com/lemon07r/Vera.git && cd Vera
cargo build --release
cp target/release/vera ~/.local/bin/
vera agent install && vera setup
```

</details>

### Updating

Vera checks for new releases once per day and prints a hint with the exact update command. To update manually:

```bash
bunx @vera-ai/cli install   # re-downloads latest binary + refreshes skill files
```

Set `VERA_NO_UPDATE_CHECK=1` to disable the automatic check.

## Model Backend

Vera itself is always local: the index lives in `.vera/`, config in `~/.vera/`. The backend choice only affects where embeddings and reranking run.

### Curated Local Models

`vera setup` downloads quantized ONNX models into `~/.vera/models/` and the ONNX Runtime shared library into `~/.vera/lib/`, then runs inference locally. No manual install required:

- **Embeddings:** [`jina-embeddings-v5-text-nano-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval) (239M, quantized ONNX). Highest-scoring embedding model under 500M parameters on MMTEB. Uses a retrieval-specific LoRA adapter designed for asymmetric search (short query, long code block).
- **Reranker:** [`jina-reranker-v2-base-multilingual`](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) (278M, quantized ONNX). Cross-encoder that scores query-document pairs jointly. Half the size and 15x faster than bge-reranker-v2-m3.

For detailed model specs, benchmarks, and training details, see [docs/models.md](docs/models.md).

With both models cached locally, the full three-stage pipeline (BM25, vector search, rerank) runs without any external calls. This gives you:

- A local repo index on disk in `.vera/`
- A local model cache under `~/.vera/models/`
- A fully self-contained setup for private repos and offline workflows

#### GPU Acceleration

By default, `vera setup` configures CPU inference. If you have a compatible GPU, use a specific backend flag:

```bash
vera setup --onnx-jina-cuda      # NVIDIA GPU (requires CUDA 12+ drivers)
vera setup --onnx-jina-rocm      # AMD GPU (Linux, requires ROCm drivers)
vera setup --onnx-jina-directml  # Any DirectX 12 GPU (Windows)
vera setup --onnx-jina-coreml    # Apple Silicon (macOS, M1/M2/M3/M4)
vera setup --onnx-jina-openvino  # Intel GPU/iGPU (Linux only, requires Intel compute runtime)
```

Vera downloads the matching ONNX Runtime build automatically. The same flag works on `vera index` and `vera search` to override the configured backend per-command.

#### Local Inference Speed

Local mode runs neural networks (239M embedding + 278M reranker) on your machine. The indexing time is compute-bound matrix math, not file I/O. These models are designed for GPU inference; CPU works but will be slow indexing a codebase for the first time. After the initial index, `vera update .` only re-embeds changed files, so subsequent updates will be fast enough even for CPU.

| Backend | Hardware | Time | Notes |
|---------|----------|------|-------|
| CUDA | RTX 4080 | **~8 s** | Recommended for large repos |
| API mode | Remote GPU | ~30 s | Requires API key, no local compute |
| CPU | Ryzen 5 7600X3D (6c/12t) | ~6 min | Use GPU or API mode if this is too slow |

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
vera search "authentication logic"
vera search "error handling" --lang rust
vera search "routes" --path "src/**/*.ts"
vera search "handler" --type function --limit 5
```

For tips on writing effective queries, filtering results, and when to use `rg` instead, see the [query guide](docs/query-guide.md).

Update the index after code changes:

```bash
vera update .
```

### Excluding Files

Vera respects `.gitignore` by default. For more control, create a `.veraignore` file in your project root using gitignore syntax. When present, `.veraignore` completely replaces `.gitignore` rules, giving you full control over what gets indexed (useful for indexing untracked local docs while excluding other files).

To keep `.gitignore` rules and add extra exclusions on top, put `#include .gitignore` at the top of `.veraignore`.

One-off exclusions without editing files:

```bash
vera index . --exclude "tests/**" --exclude "*.generated.ts"
vera update . --exclude "vendor/**"
```

Power-user flags: `--no-ignore` disables all ignore file parsing, `--no-default-excludes` disables the built-in exclusions (node_modules, .git, target, etc.).

Other useful commands:

```bash
vera doctor                    # diagnose setup issues
vera stats                     # index statistics
vera config                    # show current configuration
vera agent install             # install skill files for AI agents
vera agent status --scope all  # check skill installation status
```

Uninstall Vera and all its data:

```bash
vera uninstall
```

This removes `~/.vera/` (binary cache, models, ONNX Runtime libs, config), agent skill files, and the PATH shim. Per-project indexes (`.vera/` inside each project) are left in place. Delete them manually if needed.

If something isn't working, see [troubleshooting](docs/troubleshooting.md).

Output uses markdown codeblocks by default, the most token-efficient format for AI agents:

````
```src/auth/login.rs:42-68 function:authenticate
pub fn authenticate(credentials: &Credentials) -> Result<Token> { ... }
```
````

Use `--json` for compact single-line JSON (useful for programmatic consumption or piping to other tools), or `--raw` for verbose human-readable output with all fields. Use `--timing` to print per-stage pipeline durations (embedding, BM25, vector, fusion, reranking) to stderr.

## Benchmark Snapshot

Public comparison from `v0.4.0` (kept because it compares Vera against other tools on the same workload). Vera has improved ~55% on Recall@5 and ~83% on nDCG@10 since then. 17 tasks across `ripgrep`, `flask`, `fastify`:

| Metric | ripgrep | cocoindex-code | vector-only | Vera hybrid |
|--------|---------|----------------|-------------|-------------|
| Recall@5 | 0.2817 | 0.3730 | 0.4921 | **0.6961** |
| Recall@10 | 0.3651 | 0.5040 | 0.6627 | **0.7549** |
| MRR@10 | 0.2625 | 0.3517 | 0.2814 | **0.6009** |
| nDCG@10 | 0.2929 | 0.5206 | 0.7077 | **0.8008** |

#### Current Results (`v0.7.0+`)

21 tasks across `ripgrep`, `flask`, `fastify`, and `turborepo`:

| Version | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|--------|----------|----------|-----------|--------|---------|
| `v0.4.0` | 0.2421 | 0.5040 | 0.5159 | 0.5016 | 0.4570 |
| `v0.7.0+` | **0.7183** | **0.7778** | **0.8254** | **0.9095** | **0.8361** |

More detail: [docs/benchmarks.md](docs/benchmarks.md) · [benchmarks/indexing-performance.md](benchmarks/indexing-performance.md) · [benchmarks/reports/reproduction-guide.md](benchmarks/reports/reproduction-guide.md)

## Supported Languages

Vera supports 63 languages and file formats with tree-sitter symbol extraction, plus text chunking for data formats. Full list with extensions and extraction support: [docs/supported-languages.md](docs/supported-languages.md).

## How It Works

Vera's retrieval pipeline runs BM25 keyword search and vector similarity search in parallel, merges results with Reciprocal Rank Fusion, then reranks the top candidates with a cross-encoder. For the full breakdown (parsing, fusion, reranking, storage, and the benchmarks behind each choice), see [docs/how-it-works.md](docs/how-it-works.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for build instructions, project layout, how to add a language, and coding conventions.
