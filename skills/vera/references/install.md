# Install And First Run

## Quick Start

```sh
npx -y @vera-ai/cli install   # or: bunx @vera-ai/cli install / uvx vera-ai install
vera setup                      # downloads local ONNX models + runtime
vera index .
vera search "your query" --json
```

Combined setup + index: `vera setup --index .`

## GPU Backends

```sh
vera setup --onnx-jina-cuda       # NVIDIA GPU (CUDA 12+)
vera setup --onnx-jina-rocm       # AMD GPU (Linux, ROCm)
vera setup --onnx-jina-directml   # DirectX 12 GPU (Windows)
```

GPU flags download the matching ONNX Runtime build automatically.

## API Mode

Point Vera at any OpenAI-compatible embedding endpoint:

```sh
export EMBEDDING_MODEL_BASE_URL=https://your-embedding-api/v1
export EMBEDDING_MODEL_ID=your-model
export EMBEDDING_MODEL_API_KEY=your-key
vera setup --api
```

Optional reranker: set `RERANKER_MODEL_BASE_URL`, `RERANKER_MODEL_ID`, `RERANKER_MODEL_API_KEY` before running `vera setup --api`.

## Skill Management

```sh
vera agent install                    # install/refresh skill
vera agent install --client codex     # target a specific client
vera agent install --scope project    # project-scoped install
vera agent status --scope all         # check install status
vera agent remove --client claude     # remove from a client
```

## Diagnostics

```sh
vera doctor   # check config, models, ORT, index health
vera config   # show current config
vera stats    # index statistics
```
