# vera-ai

Code search for AI agents. Vera indexes your codebase using tree-sitter parsing and hybrid search (BM25 + vector similarity + cross-encoder reranking), then returns ranked code snippets as structured JSON.

This package downloads and wraps the native Vera binary for your platform. On musl-based Linux (Alpine, NixOS), the correct static binary is selected automatically. Set `VERA_TARGET` to override target detection (e.g., `VERA_TARGET=x86_64-unknown-linux-musl uvx vera-ai install`).

Current benchmark snapshot: on Vera's local 21-task, 4-repo release benchmark, `v0.7.0` reaches `0.78` Recall@5, `0.83` Recall@10, `0.91` MRR@10, and `0.84` nDCG@10 with the local Jina CUDA ONNX stack. Full details live in the main repo docs.

## Install

```bash
pip install vera-ai
```

## Quick Start

```bash
vera-ai setup --api
vera-ai index .
vera-ai search "authentication logic"
```

`vera-ai setup` with no flags runs an interactive wizard. `vera-ai agent install` manages skill files for your coding agents and can update `AGENTS.md` / `CLAUDE.md` style project instructions.

## Common Tasks

| Task | Command |
|------|---------|
| Use the interactive setup wizard | `vera-ai setup` |
| Use API mode (recommended) | `vera-ai setup --api` |
| Use a local NVIDIA backend | `vera-ai setup --onnx-jina-cuda` |
| Search semantically | `vera-ai search "authentication middleware"` |
| Search only changed files | `vera-ai search "authentication middleware" --changed` |
| Common structural tasks | `vera-ai structural routes` / `vera-ai structural env DATABASE_URL` |
| Find callers or callees | `vera-ai references foo` / `vera-ai references foo --callees` |
| Explain why a file is missing | `vera-ai explain-path path/to/file` |
| Inspect index health | `vera-ai stats --json` |
| Keep the index up to date | `vera-ai update .` |
| Watch for file changes | `vera-ai watch .` |
| Diagnose setup issues | `vera-ai doctor` |
| Run the deeper ONNX probe | `vera-ai doctor --probe` |
| Repair missing local assets | `vera-ai repair` |
| Inspect binary upgrades | `vera-ai upgrade` |
| Install agent skills | `vera-ai agent install` |

For the full backend matrix, model options, Docker setup, and troubleshooting, see the main [README](https://github.com/lemon07r/Vera) and [Installation Guide](https://github.com/lemon07r/Vera/blob/master/docs/installation.md).

## What you get

- **61+ languages** via tree-sitter AST parsing
- **Hybrid search**: BM25 keyword + vector similarity, fused with Reciprocal Rank Fusion
- **Cross-encoder reranking** for precision
- **Git-aware scopes and index debugging**: `--changed` / `--since` / `--base`, `explain-path`, and index health in `vera-ai stats`
- **Markdown codeblock output** by default with file paths, line ranges, and optional symbol info (use `--json` for compact JSON; `--raw` works with `vera-ai search`, `vera-ai grep`, and `vera-ai references`; `--timing` works with `vera-ai search` and `vera-ai grep`, before or after the subcommand)

For full documentation, including custom local ONNX embedding models and manual install steps, see the [GitHub repo](https://github.com/lemon07r/Vera).
