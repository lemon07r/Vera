# Installation Guide

Complete setup instructions for Vera. For the short version, see the [Quick Start](../README.md#quick-start) in the README.

## Install the Binary

Pick whichever package manager you have:

```bash
bunx @vera-ai/cli install   # Bun
npx -y @vera-ai/cli install # npm
uvx vera-ai install          # Python (uv)
```

The installer downloads the `vera` binary for your platform, writes a shim to a user bin directory, and installs global agent skill files. After that, `vera` is a standalone command.

<details>
<summary>Other install methods</summary>

**Prebuilt binaries:**
Download from [GitHub Releases](https://github.com/lemon07r/Vera/releases) for Linux (x86_64, aarch64), macOS (x86_64, aarch64), or Windows (x86_64). For Alpine, NixOS, or minimal containers without glibc, use the `x86_64-unknown-linux-musl` archive (fully static, zero runtime dependencies). The npm/pip wrappers auto-detect musl systems; to force a specific target, set `VERA_TARGET=x86_64-unknown-linux-musl` before running the install command.

**Build from source** (Rust 1.85+):
```bash
git clone https://github.com/lemon07r/Vera.git && cd Vera
cargo build --release
cp target/release/vera ~/.local/bin/
vera setup
```

**Docker** (MCP server):
```bash
docker run --rm -i -v $(pwd):/workspace ghcr.io/lemon07r/vera:cpu
```
CPU, CUDA, ROCm, and OpenVINO images available. See [docker.md](docker.md).

**Manual install:** [manual-install.md](manual-install.md)

</details>

## Set Up a Backend

Vera's index and search always run locally. The "backend" only controls where embedding and reranking models run. There are two modes:

### API Mode (recommended)

Models run on a remote server. No downloads, no GPU required, works on any hardware. You just need an API key from any OpenAI-compatible provider.

```bash
vera setup --api
```

Vera will prompt you for your endpoint URL, model ID, and API key. These get saved to Vera's config so you only enter them once.

Many providers offer free tiers or generous trial credits. Any OpenAI-compatible embedding endpoint works. Some options:

| Provider | Free tier? | Notes |
|----------|-----------|-------|
| [Jina AI](https://jina.ai/) | Yes (1M tokens free) | Vera's default local models are Jina, so the API versions work well too |
| [OpenAI](https://platform.openai.com/) | Trial credits | `text-embedding-3-small` or `text-embedding-3-large` |
| [Voyage AI](https://www.voyageai.com/) | Free tier available | Code-optimized models |
| [Cohere](https://cohere.com/) | Trial key | `embed-english-v3.0` |

You can also set the environment variables directly instead of using the wizard:

```bash
export EMBEDDING_MODEL_BASE_URL=https://api.jina.ai/v1
export EMBEDDING_MODEL_ID=jina-embeddings-v3
export EMBEDDING_MODEL_API_KEY=your-key

# Optional: reranker for better precision
export RERANKER_MODEL_BASE_URL=https://api.jina.ai/v1
export RERANKER_MODEL_ID=jina-reranker-v2-base-multilingual
export RERANKER_MODEL_API_KEY=your-key

vera setup --api
```

Only model calls leave your machine. Indexing, storage, and search remain local.

### Local Mode

Models run on your machine using ONNX Runtime. Vera downloads two small models (~100 MB total) and auto-detects your GPU. No API key needed, fully offline after setup.

**Pick the right command for your hardware:**

| You have | Command | What happens |
|----------|---------|-------------|
| Not sure | `vera setup` | Interactive wizard auto-detects your hardware |
| Apple Silicon (M1/M2/M3/M4) | `vera setup --onnx-jina-coreml` | Uses CoreML GPU acceleration |
| NVIDIA GPU | `vera setup --onnx-jina-cuda` | Uses CUDA. Fastest local option |
| AMD GPU (Linux) | `vera setup --onnx-jina-rocm` | Uses ROCm |
| Intel GPU (Linux) | `vera setup --onnx-jina-openvino` | Uses OpenVINO |
| DirectX 12 GPU (Windows) | `vera setup --onnx-jina-directml` | Uses DirectML |

> **Note:** Local mode on CPU (without a GPU) works but is slow for initial indexing. If you don't have a GPU, API mode is the better choice. After the first index, `vera update .` only re-embeds changed files, so incremental updates are fast even on CPU.

For custom ONNX models, GPU-specific tuning, and inference speed comparisons, see [models.md](models.md).

## Verify Your Setup

```bash
vera doctor          # checks config, models, and connectivity
vera doctor --probe  # deeper ONNX runtime diagnostics
```

## Index and Search

```bash
vera index .                          # index the current project
vera search "authentication logic"    # search
```

That's it. See the [query guide](query-guide.md) for tips on writing effective queries.

## Set Up Agent Skills

Vera can install skill files so your AI coding agents know how to use it:

```bash
vera agent install              # interactive: choose scope + agents
vera agent install --client all # non-interactive: all agents, global
```

This is optional but recommended if you use AI coding agents. The interactive flow can also update your project's `AGENTS.md`, `CLAUDE.md`, `COPILOT.md`, `.cursorrules`, `.clinerules`, or `.windsurfrules` file with a short Vera usage snippet.

<details>
<summary>Add the instructions manually</summary>

```markdown
## Code Search

Use Vera before opening many files or running broad text search when you need to find where logic lives or how a feature works.

- `vera search "query"` for semantic code search. Describe behavior: "JWT validation", not "auth". If one phrasing misses, try 2-3 varied queries or add `--intent "goal"`.
- `vera search ... --changed`, `--since <rev>`, or `--base <rev>` when the task is limited to modified files or a PR diff
- `vera grep "pattern"` for exact text or regex in indexed files
- `vera ast-query '<query>' --lang <lang>` for expert-level structural search with raw tree-sitter queries
- `vera explain-path path/to/file` to explain why a file is or is not indexed
- `vera references <symbol>` for callers and callees
- `vera overview` for a project summary (languages, entry points, hotspots). Add `--changed`, `--since <rev>`, or `--base <rev>` to scope it to modified files.
- `vera stats --json` for index health, including tree-sitter error, parse-failure, and Tier 0 fallback counts
- `vera search --deep "query"` for RAG-fusion query expansion + merged ranking
- Narrow `vera search` or `vera grep` with `--lang`, `--path`, `--type`, or `--scope docs`
- `vera watch .` to auto-update the index, or `vera update .` after edits (`vera index .` if `.vera/` is missing)
- For detailed usage, query patterns, and troubleshooting, read the Vera skill file installed by `vera agent install`
```

</details>

<details>
<summary>Use the Vercel skills CLI instead</summary>

```bash
npx skills add lemon07r/Vera
```

</details>

## Updating

Vera checks for new releases daily and prints a hint when one is available.

```bash
vera upgrade              # dry run: shows what would happen
vera upgrade --apply      # applies the update
```

After an upgrade, Vera automatically syncs stale agent skill installs. Set `VERA_NO_UPDATE_CHECK=1` to disable the automatic check.

If you are having trouble updating, reinstall with the package manager you originally used:

```bash
# Bun
bun install -g @vera-ai/cli && bunx @vera-ai/cli install
# npm
npm install -g @vera-ai/cli && npx @vera-ai/cli install
# uv
uvx vera-ai install
```

## Uninstalling

```bash
vera uninstall   # removes config dir, skill files, PATH shim
```

## Troubleshooting

- Run `vera doctor` to diagnose issues.
- Run `vera doctor --probe` for deeper ONNX diagnostics.
- Wrong backend? Run `vera setup` again with a different flag.
- Slow indexing on CPU? Switch to `--api` mode or use a GPU backend.
- See [troubleshooting.md](troubleshooting.md) for more.
