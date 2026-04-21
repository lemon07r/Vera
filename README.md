<div align="center">

<img width="1584" height="539" alt="vera" src="https://github.com/user-attachments/assets/c866fc70-b1e6-400b-aaf7-fa68721a4955" />

# Vera

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/lemon07r/Vera/blob/master/LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![GitHub release](https://img.shields.io/github/v/release/lemon07r/Vera?include_prereleases&sort=semver)](https://github.com/lemon07r/Vera/releases)
[![Languages](https://img.shields.io/badge/languages-65%2B-green.svg)](docs/supported-languages.md)

[Install Guide](docs/installation.md)
·
[Features](docs/features.md)
·
[Query Guide](docs/query-guide.md)
·
[Benchmarks](docs/benchmarks.md)
·
[How It Works](docs/how-it-works.md)
·
[Models](docs/models.md)
·
[Supported Languages](docs/supported-languages.md)

**V**ector **E**nhanced **R**eranking **A**gent

Code search that combines BM25 keyword matching, vector similarity, and cross-encoder reranking. Supports 65 languages (61 with tree-sitter parsing), runs locally, returns structured results with file paths, line ranges, symbol metadata, and relevance scores.

</div>

## Quick Start

**1. Install**
```bash
bunx @vera-ai/cli install   # or: npx -y @vera-ai/cli install / uvx vera-ai install
```

**2. Set up models** (pick one)
```bash
vera setup                       # Interactive wizard (auto-detects your hardware)
vera setup --api                 # API mode: works everywhere, no GPU needed (recommended)
vera setup --onnx-jina-coreml    # Apple Silicon (M1/M2/M3/M4)
vera setup --onnx-jina-cuda      # NVIDIA GPU
vera setup --onnx-jina-rocm      # AMD GPU (ROCm, Linux)
vera setup --onnx-jina-openvino  # Intel GPU (OpenVINO, Linux)
vera setup --onnx-jina-directml  # DirectX 12 GPU (Windows)
```

**3. Index and search**
```bash
vera index .
vera search "authentication logic"
```

## What Sets Vera Apart

| | |
|---|---|
| **Cross-encoder reranking** | Most tools stop at retrieval. Vera scores query-candidate pairs jointly, lifting MRR@10 from 0.28 to 0.60. |
| **Single binary, 65 languages** | One static binary with 61 tree-sitter grammars compiled in. No Python, no language servers, no per-language toolchains. |
| **Built-in code intelligence** | Call graph analysis, reference finding, dead code detection, and project overview, all from the same index. |
| **Token-efficient for agents** | Returns symbol-bounded chunks, not entire files. 75-95% fewer tokens on typical queries. |

Vera started after weeks of working on Pampax, a project I forked because it and other similar tools were missing what I wanted. I kept running into deep-rooted bugs, less-than-ideal design decisions, and thought I could build something better from the ground up. Every design choice comes from careful research, learning from other projects, benchmarking and evaluation. Take a look at the full [feature list](docs/features.md) to see everything Vera can do.

## Installation

Use the quick start above if you just want to get going. This section helps you pick the right backend.

```bash
bunx @vera-ai/cli install   # or: npx -y @vera-ai/cli install / uvx vera-ai install
```

### Pick Your Backend

Vera itself is always local: the index lives in `.vera/` per project, config and models in `$XDG_DATA_HOME/vera` (or `~/.vera` for existing installs). The backend choice only affects where embeddings and reranking run.

| You have | Run this | What happens |
|----------|----------|-------------|
| Not sure | `vera setup` | Interactive wizard auto-detects your hardware |
| Any hardware | `vera setup --api` | Models run remotely via any OpenAI-compatible API. No GPU needed. **Recommended.** |
| Apple Silicon (M1/M2/M3/M4) | `vera setup --onnx-jina-coreml` | Downloads local models, uses CoreML GPU acceleration |
| NVIDIA GPU | `vera setup --onnx-jina-cuda` | Downloads local models, uses CUDA. Fastest local option |
| AMD GPU (Linux) | `vera setup --onnx-jina-rocm` | Downloads local models, uses ROCm |
| Intel GPU (Linux) | `vera setup --onnx-jina-openvino` | Downloads local models, uses OpenVINO |
| DirectX 12 GPU (Windows) | `vera setup --onnx-jina-directml` | Downloads local models, uses DirectML |

API mode works with any OpenAI-compatible endpoint and needs no local compute. Local mode downloads two curated ONNX models and auto-detects your GPU; a GPU is recommended since CPU-only indexing is slow. After the first index, `vera update .` only re-embeds changed files, so incremental updates are fast on any backend. Full details: [docs/models.md](docs/models.md).

For step-by-step instructions, API provider options, Docker, building from source, and troubleshooting, see the full [Installation Guide](docs/installation.md).

<details>
<summary>MCP server</summary>

```bash
vera mcp   # or: bunx @vera-ai/cli mcp / uvx vera-ai mcp
```
Exposes `search_code`, `get_stats`, `get_overview`, `regex_search`, `structural_search`, and `explain_path` tools. `search_code` and `structural_search` auto-index and start a file watcher on first use if no index exists.
The MCP surface stays intentionally small; use the CLI skill path when you need the full command set.

</details>

## Usage

### Core Workflow

```bash
vera index .
vera search "authentication logic"
vera update .
```

### Search Patterns

```bash
vera search "error handling" --lang rust
vera search "routes" --path "src/**/*.ts"
vera search "handler" --type function --limit 5
vera search "OAuth token refresh" "JWT expiry handling" "auth middleware"
vera search "config" --intent "find where database connection strings are loaded"
vera search "config loading" --deep
vera search "auth" --compact
vera search "token validation" --changed
vera search "config loading" --base origin/main
vera structural definitions parse_config
vera structural env DATABASE_URL
vera structural routes --path "src/**/*.ts"
vera ast-query '(function_item name: (identifier) @fn)' --lang rust
```

### Common Tasks

| Task | Command |
|------|---------|
| Regex or exact text | `vera grep "fn\s+main"` |
| Common structural tasks | `vera structural routes` / `vera structural env DATABASE_URL` |
| Explain why a file is missing from the index | `vera explain-path path/to/file` |
| Structural AST search | `vera ast-query '(function_item name: (identifier) @fn)' --lang rust` |
| Inspect index health | `vera stats --json` |
| Find callers | `vera references foo` or `vera structural calls foo` |
| Find callees | `vera references foo --callees` |
| Find dead code | `vera dead-code` |
| Get a project overview | `vera overview` |
| Scope a search to changed files | `vera search "query" --changed` |
| Keep the index fresh | `vera watch .` |
| Check your setup | `vera doctor` |
| Repair missing local assets | `vera repair` |
| Install agent skills | `vera agent install` |

See the [query guide](docs/query-guide.md) for search tips, the [feature list](docs/features.md) for the full command surface, and `vera --help` for CLI details.

### Output

Defaults to markdown codeblocks (the most token-efficient format for AI agents):

````
```src/auth/login.rs:42-68 function:authenticate
pub fn authenticate(credentials: &Credentials) -> Result<Token> { ... }
```
````

Use `--json` for compact JSON. `--raw` and `--timing` work with `vera search` and `vera grep`, and you can place them before or after the subcommand (for example, `vera --timing search "auth"` or `vera grep "TODO" --raw`).

### Excluding Files

Vera respects `.gitignore` by default. Create a `.veraignore` file (gitignore syntax) for more control, or use `--exclude` flags. Details: [docs/features.md](docs/features.md#flexible-exclusions).

If a file is missing from the index and you need the exact reason, run:

```bash
vera explain-path path/to/file
```

## Benchmarks

21-task benchmark across `ripgrep`, `flask`, `fastify`, and `turborepo`:

| Metric | ripgrep | cocoindex | ColGREP (149M) | Vera |
|--------|---------|-----------|----------------|------|
| Recall@5 | 0.28 | 0.37 | 0.67 | **0.78** |
| MRR@10 | 0.26 | 0.35 | 0.62 | **0.91** |
| nDCG@10 | 0.29 | 0.52 | 0.56 | **0.84** |

Full methodology and version history: [docs/benchmarks.md](docs/benchmarks.md).

## Configure Your AI Agent

`vera agent install` installs the Vera skill for supported coding agents and can add a short usage snippet to your project's `AGENTS.md`, `CLAUDE.md`, `COPILOT.md`, or editor rules file.

```bash
vera agent install
vera agent install --client all
```

If you use the [skills CLI](https://github.com/vercel-labs/skills), you can install Vera there too:

```bash
npx skills add lemon07r/Vera
```

If you skipped the prompt and want to add the instructions manually, use the snippet in the [Installation Guide](docs/installation.md#set-up-agent-skills).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
