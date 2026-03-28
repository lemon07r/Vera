# @vera-ai/cli

Code search for AI agents. Vera indexes your codebase using tree-sitter parsing and hybrid search (BM25 + vector similarity + cross-encoder reranking), then returns ranked code snippets as structured JSON.

This package downloads and wraps the native Vera binary for your platform.

Current benchmark snapshot: on Vera's local 21-task, 4-repo release benchmark, `v0.7.0` reaches `0.78` Recall@5, `0.83` Recall@10, `0.91` MRR@10, and `0.84` nDCG@10 with the local Jina CUDA ONNX stack. Full details live in the main repo docs.

## Install

```bash
npm install -g @vera-ai/cli
```

`vera setup` only configures the backend. Run `vera agent install` to set up skill files for your agents (interactive by default, or pass `--client` and `--scope` for non-interactive use).

## Usage

```bash
# Optional: install skill files for your agents
vera agent install

# Index a project
vera index .

# Search
vera search "authentication middleware"

# Local ONNX inference (no API keys needed. downloads models automatically)
vera index . --onnx-jina-cpu
vera search "error handling" --onnx-jina-cpu

# Optional local CodeRankEmbed preset
vera setup --code-rank-embed --onnx-jina-cuda

# GPU acceleration (NVIDIA/AMD/DirectML/CoreML/OpenVINO)
vera index . --onnx-jina-cuda

# Diagnose or repair local setup issues
vera doctor
vera doctor --probe
vera repair
vera upgrade
```

`vera doctor --probe` runs a deeper read-only ONNX session check. `vera upgrade` shows the binary update plan and can apply it when the install method is known.

On GPU backends, Vera uses a free-VRAM-aware batch ceiling and sequence-aware local micro-batching, and it reuses learned device-specific batch windows across runs.

## What you get

- **60+ languages** via tree-sitter AST parsing
- **Hybrid search**: BM25 keyword + vector similarity, fused with Reciprocal Rank Fusion
- **Cross-encoder reranking** for precision
- **Markdown codeblock output** by default with file paths, line ranges, and optional symbol info (use `--json` for compact JSON, `--raw` for verbose output, `--timing` for step durations)

For full documentation, including custom local ONNX embedding models and manual install steps, see the [GitHub repo](https://github.com/lemon07r/Vera).
