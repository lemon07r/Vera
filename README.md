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

After trying many other tools and maintaining Pampax, a fork of someone's code search tool, I ran into constant issues. The upstream project was hastily thrown together with deep-rooted bugs. Despite significantly improving Pampax over time, the foundation was too fragile to build on reliably. Nothing supported all the things I wanted (like provider-agnostic reranking), so I built something better from scratch. Every design choice in Vera (the retrieval pipeline, the model selection, the output format) comes from hours of research, real benchmarking and evaluation.

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

**Cross-encoder reranking.** Most code search tools retrieve candidates and stop. Vera adds a reranking stage that reads query and candidate as a single pair, scoring relevance jointly instead of comparing pre-computed vectors. Result: 0.60 MRR@10 vs. 0.28 with vector retrieval alone.

**Zero-dependency, single binary.** One static binary with 60+ tree-sitter grammars compiled in. No Python, no language servers, no per-language toolchains. Drop it on any machine, run `vera setup`, done. Compare: Serena requires Python, uv, and separate LSP installs per language.

**Higher accuracy, proven on real codebases.** Vera scores 0.80 nDCG@10 and 0.70 Recall@5 on a 17-task benchmark across `ripgrep`, `flask`, and `fastify`. The current 21-task suite scores even higher. See [Benchmark Snapshot](#benchmark-snapshot).

**Token-efficient output.** Defaults to markdown codeblocks, cutting output size ~35-40% vs. JSON. Ships with skill files that teach AI agents how to write effective queries and when to reach for `rg` instead.

## Features

**Model-agnostic, local-first.** Point Vera at any OpenAI-compatible embedding or reranker endpoint, remote or local. Indexing, storage, and search logic always stay on your machine. Run `vera setup` to download two curated ONNX models and run the full pipeline offline. Details: [Model Backend](#model-backend).

**Tree-sitter structural parsing.** 60+ language grammars extract functions, classes, methods, and structs as discrete chunks. Results map to actual symbol boundaries, not arbitrary line ranges. Filter with `--type function` or `--type class`.

**Structured, code-aware results.** Every result includes file path, line range, source content, symbol name, and type. Agents and scripts consume this directly without parsing. See [AGENT-USAGE.md](AGENT-USAGE.md) for AI agent integration.

## Installation

```bash
bunx @vera-ai/cli install   # or: npx -y @vera-ai/cli install / uvx vera-ai install
```

This downloads the `vera` binary, adds it to your PATH, and installs agent skill files. After this, `vera` is a standalone command. You don't need `bunx`/`npx`/`uvx` again.

```bash
vera setup          # interactive backend menu (auto-detects your GPU)
vera index .        # index the current project (creates .vera/ in project root)
vera search "query" # search (each project gets its own index)
vera update .       # after code changes
```

`vera setup` with no flags shows a backend picker and auto-detects your GPU. You can also skip the menu: `vera setup --onnx-jina-cuda` (NVIDIA), `--onnx-jina-coreml` (Apple Silicon), `--api` (remote endpoints), etc. Run `vera setup --help` for all options.

Use `vera doctor` if anything goes wrong. It reports the saved and active backend, installed Vera version, and checks GitHub for newer releases. Add `--probe` for a deeper read-only ONNX session check that does not download or repair missing assets. Use `vera repair` to re-fetch missing local assets or re-save API config from the current environment. Use `vera upgrade` to inspect or apply the binary update plan.

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

Vera checks for new releases once per day and prints a hint when a newer release is available. To inspect the current update plan:

```bash
vera upgrade
vera upgrade --apply
```

`vera upgrade` is a dry run by default. It shows the detected install method and the exact command Vera would run. `--apply` only runs when Vera can determine a single install method.

You can still update manually:

```bash
bunx @vera-ai/cli install   # re-downloads latest binary + refreshes skill files
```

Set `VERA_NO_UPDATE_CHECK=1` to disable the automatic check.

## Model Backend

Vera itself is always local: the index lives in `.vera/`, config in `~/.vera/`. The backend choice only affects where embeddings and reranking run.

### Curated Local Models

`vera setup` downloads quantized ONNX models into `~/.vera/models/` and the ONNX Runtime library into `~/.vera/lib/`. No manual install required.

| Model | Size | Role |
|-------|------|------|
| [`jina-embeddings-v5-text-nano-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval) | 239M | Embedding (retrieval LoRA, asymmetric search) |
| [`jina-reranker-v2-base-multilingual`](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | 278M | Cross-encoder reranker (15x faster than bge-reranker-v2-m3) |

With both models cached, the full pipeline (BM25 + vector search + rerank) runs without external calls. For detailed specs, see [docs/models.md](docs/models.md).

### GPU Acceleration

`vera setup` auto-detects your GPU. You can also specify a backend directly:

| Flag | Hardware |
|------|----------|
| `--onnx-jina-cuda` | NVIDIA (CUDA 12+) |
| `--onnx-jina-rocm` | AMD (Linux, ROCm) |
| `--onnx-jina-directml` | Any DirectX 12 GPU (Windows) |
| `--onnx-jina-coreml` | Apple Silicon (macOS) |
| `--onnx-jina-openvino` | Intel GPU/iGPU (Linux) |

Vera downloads the matching ONNX Runtime build automatically. For OpenVINO and ROCm (no pre-built binaries on GitHub), Vera installs via pip into a managed venv at `~/.vera/venv/`, falling back to direct PyPI wheel download if pip is unavailable. The same flag works on `vera index` and `vera search` to override the configured backend per-command.

### Inference Speed

Local mode runs neural networks on your machine. GPU is recommended; CPU works but is slow for initial indexing. After the first index, `vera update .` only re-embeds changed files, so updates are fast even on CPU.

| Backend | Hardware | Time | Notes |
|---------|----------|------|-------|
| CUDA | RTX 4080 | **~8 s** | Recommended for large repos |
| API mode | Remote GPU | ~30 s | Requires API key, no local compute |
| CPU | Ryzen 5 7600X3D (6c/12t) | ~6 min | Use GPU or API mode if this is too slow |

### API Mode

Use `vera setup --api` to point Vera at any OpenAI-compatible endpoint (remote APIs or local servers like `llama.cpp`).

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

See the [query guide](docs/query-guide.md) for tips on writing effective queries and when to use `rg` instead.

Update the index after code changes: `vera update .`

### Excluding Files

Vera respects `.gitignore` by default. For more control, create a `.veraignore` file in your project root (gitignore syntax). When present, `.veraignore` completely replaces `.gitignore` rules, giving you full control over what gets indexed.

To keep `.gitignore` rules and add extra exclusions on top, put `#include .gitignore` at the top of `.veraignore`.

One-off exclusions:

```bash
vera index . --exclude "tests/**" --exclude "*.generated.ts"
vera update . --exclude "vendor/**"
```

`--no-ignore` disables all ignore file parsing. `--no-default-excludes` disables built-in exclusions (node_modules, .git, target, etc.).

### Output Format

Defaults to markdown codeblocks (the most token-efficient format for AI agents):

````
```src/auth/login.rs:42-68 function:authenticate
pub fn authenticate(credentials: &Credentials) -> Result<Token> { ... }
```
````

| Flag | Output |
|------|--------|
| `--json` | Compact single-line JSON |
| `--raw` | Verbose human-readable output |
| `--timing` | Per-stage pipeline durations to stderr |

### Other Commands

```bash
vera doctor                    # diagnose setup issues
vera doctor --probe            # deeper read-only ONNX probe
vera doctor --probe --json     # machine-readable deep diagnostics
vera repair                    # re-fetch missing assets for current backend
vera upgrade                   # inspect the binary update plan
vera upgrade --apply           # run it when the install method is known
vera stats                     # index statistics
vera config                    # show current configuration
vera agent install             # install skill files for AI agents
vera agent status --scope all  # check skill installation status
```

### Uninstalling

```bash
vera uninstall
```

Removes `~/.vera/` (binary, models, ONNX Runtime libs, config), agent skill files, and the PATH shim. Per-project indexes (`.vera/` in each project) are left in place.

If something isn't working, see [troubleshooting](docs/troubleshooting.md).

## Benchmark Snapshot

Comparison from `v0.4.0` against other tools on the same workload (17 tasks across `ripgrep`, `flask`, `fastify`). Vera has improved ~55% on Recall@5 and ~83% on nDCG@10 since this comparison.

| Metric | ripgrep | cocoindex-code | vector-only | Vera hybrid |
|--------|---------|----------------|-------------|-------------|
| Recall@5 | 0.2817 | 0.3730 | 0.4921 | **0.6961** |
| Recall@10 | 0.3651 | 0.5040 | 0.6627 | **0.7549** |
| MRR@10 | 0.2625 | 0.3517 | 0.2814 | **0.6009** |
| nDCG@10 | 0.2929 | 0.5206 | 0.7077 | **0.8008** |

### Current Results (v0.7.0+)

21 tasks across `ripgrep`, `flask`, `fastify`, and `turborepo`:

| Version | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|--------|----------|----------|-----------|--------|---------|
| `v0.4.0` | 0.2421 | 0.5040 | 0.5159 | 0.5016 | 0.4570 |
| `v0.7.0+` | **0.7183** | **0.7778** | **0.8254** | **0.9095** | **0.8361** |

More detail: [docs/benchmarks.md](docs/benchmarks.md) · [benchmarks/indexing-performance.md](benchmarks/indexing-performance.md) · [benchmarks/reports/reproduction-guide.md](benchmarks/reports/reproduction-guide.md)

## Supported Languages

63 languages and file formats with tree-sitter symbol extraction, plus text chunking for data formats. Full list: [docs/supported-languages.md](docs/supported-languages.md).

## How It Works

BM25 keyword search and vector similarity run in parallel, merge via Reciprocal Rank Fusion, then a cross-encoder reranks the top candidates. Full breakdown: [docs/how-it-works.md](docs/how-it-works.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for build instructions, project layout, how to add a language, and coding conventions.
