<div align="center">

<img width="1584" height="539" alt="vera" src="https://github.com/user-attachments/assets/c866fc70-b1e6-400b-aaf7-fa68721a4955" />

# Vera

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/lemon07r/Vera/blob/master/LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![GitHub release](https://img.shields.io/github/v/release/lemon07r/Vera?include_prereleases&sort=semver)](https://github.com/lemon07r/Vera/releases)
[![Languages](https://img.shields.io/badge/languages-64%2B-green.svg)](docs/supported-languages.md)

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

Code search that combines BM25 keyword matching, vector similarity, and cross-encoder reranking. Supports 64 languages (60 with tree-sitter parsing), runs locally, returns structured results with file paths, line ranges, symbol metadata, and relevance scores.

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
| **Single binary, 64 languages** | One static binary with 60 tree-sitter grammars compiled in. No Python, no language servers, no per-language toolchains. |
| **Built-in code intelligence** | Call graph analysis, reference finding, dead code detection, and project overview, all from the same index. |
| **Token-efficient for agents** | Returns symbol-bounded chunks, not entire files. 75-95% fewer tokens on typical queries. |

Vera started after weeks of working on Pampax, a project I forked because it and other similar tools were missing what I wanted. I kept running into deep-rooted bugs and less-than-ideal design decisions, and realized starting fresh would be better. Every design choice comes from careful research, learning from other projects, benchmarking and evaluation. Take a look at the full [feature list](docs/features.md) to see everything Vera can do.

## Installation

```bash
bunx @vera-ai/cli install   # or: npx -y @vera-ai/cli install / uvx vera-ai install
```

### Pick Your Backend

| You have | Run this | What happens |
|----------|----------|-------------|
| Not sure | `vera setup` | Interactive wizard auto-detects your hardware |
| Any hardware | `vera setup --api` | Models run remotely via any OpenAI-compatible API. No GPU needed. **Recommended.** |
| Apple Silicon (M1/M2/M3/M4) | `vera setup --onnx-jina-coreml` | Downloads local models, uses CoreML GPU acceleration |
| NVIDIA GPU | `vera setup --onnx-jina-cuda` | Downloads local models, uses CUDA. Fastest local option |
| AMD GPU (Linux) | `vera setup --onnx-jina-rocm` | Downloads local models, uses ROCm |
| Intel GPU (Linux) | `vera setup --onnx-jina-openvino` | Downloads local models, uses OpenVINO |
| DirectX 12 GPU (Windows) | `vera setup --onnx-jina-directml` | Downloads local models, uses DirectML |

After setup, indexing and search work the same regardless of backend.

For step-by-step instructions, API provider options, Docker, building from source, and troubleshooting, see the full [Installation Guide](docs/installation.md).

<details>
<summary>MCP server</summary>

```bash
vera mcp   # or: bunx @vera-ai/cli mcp / uvx vera-ai mcp
```
Exposes `search_code`, `get_stats`, `get_overview`, and `regex_search` tools. `search_code` auto-indexes and starts a file watcher on first use if no index exists.

</details>

## Usage

```bash
vera search "authentication logic"
vera search "error handling" --lang rust
vera search "routes" --path "src/**/*.ts"
vera search "handler" --type function --limit 5
vera search "config loading" --deep              # RAG-fusion query expansion (or iterative symbol-following)
vera search "auth" --compact                     # signatures only, broad exploration
```

Update the index after code changes: `vera update .`

Keep the index fresh automatically: `vera watch .`

See the [query guide](docs/query-guide.md) for tips on writing effective queries.

<details>
<summary>More commands</summary>

```bash
# Code intelligence
vera grep "fn\s+main"              # regex search over indexed files
vera references foo                # find all callers of symbol 'foo'
vera references foo --callees      # find what 'foo' calls
vera dead-code                     # find functions with no callers
vera overview                      # project summary: languages, entry points, hotspots

# Index management
vera index .                       # index current project
vera update .                      # re-index changed files
vera watch .                       # auto-update on file changes (Ctrl-C to stop)
vera stats                         # index statistics

# Setup and diagnostics
vera doctor                        # diagnose setup issues
vera doctor --probe                # deeper read-only ONNX probe
vera repair                        # re-fetch missing assets
vera upgrade                       # inspect binary update plan
vera config                        # show current configuration
vera backend                       # manage ONNX runtime and model backend

# Agent skills
vera agent install                 # interactive: choose scope + agents
vera agent install --client all    # non-interactive: all agents, global
vera agent status                  # check skill installation status
vera agent sync                    # refresh stale skill installs
vera agent remove                  # pick installs to remove

# Cleanup
vera uninstall                     # removes config dir, skill files, PATH shim
```

</details>

### Output Format

Defaults to markdown codeblocks (the most token-efficient format for AI agents):

````
```src/auth/login.rs:42-68 function:authenticate
pub fn authenticate(credentials: &Credentials) -> Result<Token> { ... }
```
````

Use `--json` for compact JSON, `--raw` for verbose human-readable output, or `--timing` for per-stage pipeline durations.

### Excluding Files

Vera respects `.gitignore` by default. Create a `.veraignore` file (gitignore syntax) for more control, or use `--exclude` flags. Details: [docs/features.md](docs/features.md#flexible-exclusions).

## Model Backend

Vera itself is always local: the index lives in `.vera/` per project, config and models in `$XDG_DATA_HOME/vera` (or `~/.vera` for existing installs). The backend choice only affects where embeddings and reranking run.

API mode works with any OpenAI-compatible endpoint and needs no local compute. Local mode downloads two curated ONNX models and auto-detects your GPU; a GPU is recommended for local mode since CPU-only indexing is slow. After the first index, `vera update .` only re-embeds changed files, so incremental updates are fast on any backend.

Full details: [docs/models.md](docs/models.md).

## Benchmarks

21-task benchmark across `ripgrep`, `flask`, `fastify`, and `turborepo`:

| Metric | ripgrep | cocoindex | ColGREP (149M) | Vera |
|--------|---------|-----------|----------------|------|
| Recall@5 | 0.28 | 0.37 | 0.67 | **0.78** |
| MRR@10 | 0.26 | 0.35 | 0.62 | **0.91** |
| nDCG@10 | 0.29 | 0.52 | 0.56 | **0.84** |

Full methodology and version history: [docs/benchmarks.md](docs/benchmarks.md).

## Configure Your AI Agent

`vera agent install` installs the Vera skill for your coding agents and offers to add a usage snippet to your project's `AGENTS.md`, `CLAUDE.md`, `COPILOT.md`, or editor rules file. Installed agents start preselected in the interactive picker, deselecting one removes its existing Vera skill install, and stale installs can be refreshed in one step before you enter the full picker.

Alternatively, install the Vera skill with the [skills CLI](https://github.com/vercel-labs/skills):

```bash
npx skills add lemon07r/Vera
```

If you skipped the prompt or want to add it manually:

```markdown
## Code Search

Use Vera before opening many files or running broad text search when you need to find where logic lives or how a feature works.

- `vera search "query"` for semantic code search. Describe behavior: "JWT validation", not "auth".
- `vera grep "pattern"` for exact text or regex
- `vera references <symbol>` for callers and callees
- `vera overview` for a project summary (languages, entry points, hotspots)
- `vera search --deep "query"` for RAG-fusion query expansion + merged ranking
- Narrow results with `--lang`, `--path`, `--type`, or `--scope docs`
- `vera watch .` to auto-update the index, or `vera update .` after edits (`vera index .` if `.vera/` is missing)
- For detailed usage, query patterns, and troubleshooting, read the Vera skill file installed by `vera agent install`
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
