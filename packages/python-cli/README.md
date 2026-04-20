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

`vera-ai setup` only configures the backend. Run `vera-ai agent install` to set up skill files for your agents. The interactive flow can also update `AGENTS.md` / `CLAUDE.md` style project instructions for you.

## Common Tasks

| Task | Command |
|------|---------|
| Use API mode (recommended) | `vera-ai setup --api` |
| Use the interactive setup wizard | `vera-ai setup` |
| Use a local NVIDIA backend | `vera-ai setup --onnx-jina-cuda` |
| Search semantically | `vera-ai search "authentication middleware"` |
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
- **Markdown codeblock output** by default with file paths, line ranges, and optional symbol info (use `--json` for compact JSON; `--raw` and `--timing` work with `vera search` and `vera grep`, before or after the subcommand)

For full documentation, including custom local ONNX embedding models and manual install steps, see the [GitHub repo](https://github.com/lemon07r/Vera).
