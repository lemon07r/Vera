<div align="center">

<img width="1584" height="539" alt="vera" src="https://github.com/user-attachments/assets/c866fc70-b1e6-400b-aaf7-fa68721a4955" />

# Vera

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/lemon07r/Vera/blob/master/LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![GitHub release](https://img.shields.io/github/v/release/lemon07r/Vera?include_prereleases&sort=semver)](https://github.com/lemon07r/Vera/releases)
[![Languages](https://img.shields.io/badge/languages-63%2B-green.svg)](docs/supported-languages.md)

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

Code search that combines BM25 keyword matching, vector similarity, and cross-encoder reranking. Supports 63 languages (58 with tree-sitter parsing), runs locally, returns structured results with file paths, line ranges, symbol metadata, and relevance scores.

</div>

## Quick Start

```bash
bunx @vera-ai/cli install   # or: npx -y @vera-ai/cli install / uvx vera-ai install
vera setup                   # downloads local models, no API needed
vera index .
vera search "authentication logic"
```

## What Sets Vera Apart

| | |
|---|---|
| **Cross-encoder reranking** | Most tools stop at retrieval. Vera scores query-candidate pairs jointly, lifting MRR@10 from 0.28 to 0.60. |
| **Single binary, 63 languages** | One static binary with 58 tree-sitter grammars compiled in. No Python, no language servers, no per-language toolchains. |
| **Built-in code intelligence** | Call graph analysis, reference finding, dead code detection, and project overview, all from the same index. |
| **Token-efficient for agents** | Returns symbol-bounded chunks, not entire files. 75-95% fewer tokens on typical queries. |

Vera started after months of maintaining Pampax (a fork of someone's code search tool) and running into deep-rooted bugs and missing features like provider-agnostic reranking. Every design choice comes from real benchmarking and evaluation. See the full [feature list](docs/features.md) for everything Vera can do.

## Installation

```bash
bunx @vera-ai/cli install   # or: npx -y @vera-ai/cli install / uvx vera-ai install
vera setup
```

The installer downloads the `vera` binary, writes a shim to a user bin directory, and installs global agent skill files. After that, `vera` is a standalone command.

`vera setup` runs an interactive wizard for backend selection, agent skill installation, and optional project indexing. Skip the wizard with flags: `--onnx-jina-cuda` (NVIDIA), `--onnx-jina-coreml` (Apple Silicon), `--api` (remote endpoints). Run `vera setup --help` for all options.

<details>
<summary>Other install methods</summary>

**MCP server** (JSON-RPC over stdio):
```bash
vera mcp   # or: bunx @vera-ai/cli mcp / uvx vera-ai mcp
```
Exposes `search_code`, `index_project`, `update_project`, `get_stats`, `get_overview`, `watch_project`, `find_references`, `find_dead_code`, and `regex_search` tools.

**Docker** (MCP server):
```bash
docker run --rm -i -v $(pwd):/workspace ghcr.io/lemon07r/vera:cpu
```
CPU, CUDA, ROCm, and OpenVINO images available. See [docs/docker.md](docs/docker.md).

**Prebuilt binaries:**
Download from [GitHub Releases](https://github.com/lemon07r/Vera/releases) for Linux (x86_64, aarch64), macOS (x86_64, aarch64), or Windows (x86_64).

**Build from source** (Rust 1.85+):
```bash
git clone https://github.com/lemon07r/Vera.git && cd Vera
cargo build --release
cp target/release/vera ~/.local/bin/
vera setup
```

**Manual install:** [docs/manual-install.md](docs/manual-install.md)

</details>

<details>
<summary>Updating</summary>

Vera checks for new releases daily and prints a hint when one is available.

```bash
vera upgrade              # dry run: shows what would happen
vera upgrade --apply      # applies the update
```

After an upgrade, Vera automatically syncs stale agent skill installs. You can also re-run the installer: `bunx @vera-ai/cli install`. Set `VERA_NO_UPDATE_CHECK=1` to disable the automatic check.

</details>

## Usage

```bash
vera search "authentication logic"
vera search "error handling" --lang rust
vera search "routes" --path "src/**/*.ts"
vera search "handler" --type function --limit 5
vera search "config loading" --deep              # follows symbols from initial results
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
vera uninstall                     # removes ~/.vera/, skill files, PATH shim
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

Vera itself is always local: the index lives in `.vera/`, config in `~/.vera/`. The backend choice only affects where embeddings and reranking run.

`vera setup` downloads two curated ONNX models and auto-detects your GPU. GPU is recommended; CPU works but is slow for initial indexing. After the first index, `vera update .` only re-embeds changed files, so updates are fast even on CPU.

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

`vera agent install` installs the Vera skill for your coding agents and offers to add a usage snippet to your project's `AGENTS.md` (or `CLAUDE.md`, `.cursorrules`, etc.).

If you skipped the prompt or want to add it manually:

```markdown
## Code Search

This project is indexed with Vera. Use `vera search "query"` for semantic code search
and `vera grep "pattern"` for regex search. Run `vera update .` after code changes.
For query tips and output format details, see the Vera skill in your skills directory.
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
