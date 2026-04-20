# @vera-ai/cli

Code search for AI agents. Vera indexes your codebase using tree-sitter parsing and hybrid search (BM25 + vector similarity + cross-encoder reranking), then returns ranked code snippets as structured JSON.

This package downloads and wraps the native Vera binary for your platform. On musl-based Linux (Alpine, NixOS), the correct static binary is selected automatically. Set `VERA_TARGET` to override target detection (e.g., `VERA_TARGET=x86_64-unknown-linux-musl npm install -g @vera-ai/cli`).

Current benchmark snapshot: on Vera's local 21-task, 4-repo release benchmark, `v0.7.0` reaches `0.78` Recall@5, `0.83` Recall@10, `0.91` MRR@10, and `0.84` nDCG@10 with the local Jina CUDA ONNX stack. Full details live in the main repo docs.

## Install

```bash
npm install -g @vera-ai/cli
```

## Quick Start

```bash
vera setup --api
vera index .
vera search "authentication logic"
```

`vera setup` with no flags runs an interactive wizard. `vera agent install` manages skill files for your coding agents and can update `AGENTS.md` / `CLAUDE.md` style project instructions.

## Common Tasks

| Task | Command |
|------|---------|
| Use the interactive setup wizard | `vera setup` |
| Use API mode (recommended) | `vera setup --api` |
| Use a local NVIDIA backend | `vera setup --onnx-jina-cuda` |
| Search semantically | `vera search "authentication middleware"` |
| Keep the index up to date | `vera update .` |
| Watch for file changes | `vera watch .` |
| Diagnose setup issues | `vera doctor` |
| Run the deeper ONNX probe | `vera doctor --probe` |
| Repair missing local assets | `vera repair` |
| Inspect binary upgrades | `vera upgrade` |
| Install agent skills | `vera agent install` |

For the full backend matrix, model options, Docker setup, and troubleshooting, see the main [README](https://github.com/lemon07r/Vera) and [Installation Guide](https://github.com/lemon07r/Vera/blob/master/docs/installation.md).

## What you get

- **61+ languages** via tree-sitter AST parsing
- **Hybrid search**: BM25 keyword + vector similarity, fused with Reciprocal Rank Fusion
- **Cross-encoder reranking** for precision
- **Markdown codeblock output** by default with file paths, line ranges, and optional symbol info (use `--json` for compact JSON; `--raw` and `--timing` work with `vera search` and `vera grep`, before or after the subcommand)

For full documentation, including custom local ONNX embedding models and manual install steps, see the [GitHub repo](https://github.com/lemon07r/Vera).
