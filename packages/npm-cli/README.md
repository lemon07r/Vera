# @vera-ai/cli

Code search for AI agents. Vera indexes your codebase using tree-sitter parsing and hybrid search (BM25 + vector similarity + cross-encoder reranking), then returns ranked code snippets as structured JSON.

This package downloads and wraps the native Vera binary for your platform.

Current benchmark snapshot: on Vera's local 21-task, 4-repo release benchmark, `v0.7.0` reaches `0.78` Recall@5, `0.83` Recall@10, `0.91` MRR@10, and `0.84` nDCG@10 with the local Jina CUDA ONNX stack. Full details live in the main repo docs.

## Install

```bash
npm install -g @vera-ai/cli
```

## Usage

```bash
# Index a project
vera index .

# Search
vera search "authentication middleware"

# Local ONNX inference (no API keys needed. downloads models automatically)
vera index . --onnx-jina-cpu
vera search "error handling" --onnx-jina-cpu

# GPU acceleration (NVIDIA/AMD/DirectML)
vera index . --onnx-jina-cuda
```

## What you get

- **60+ languages** via tree-sitter AST parsing
- **Hybrid search**: BM25 keyword + vector similarity, fused with Reciprocal Rank Fusion
- **Cross-encoder reranking** for precision
- **Compact JSON output** with file paths, line ranges, code content, and optional symbol info (use `--raw` for verbose output)

For full documentation, see the [GitHub repo](https://github.com/lemon07r/Vera).
