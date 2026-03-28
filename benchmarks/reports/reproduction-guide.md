# Benchmark Reproduction Guide

This guide reproduces the public benchmark snapshot referenced in [README.md](../../README.md) and [docs/benchmarks.md](../../docs/benchmarks.md).

## Requirements

- Linux x86_64
- Rust toolchain installed
- Python 3
- `git`
- network access for embedding and reranking endpoints

## API Configuration

Create `secrets.env` at the repository root:

```bash
EMBEDDING_MODEL_BASE_URL=<your-embedding-api-url>
EMBEDDING_MODEL_ID=Qwen/Qwen3-Embedding-8B
EMBEDDING_MODEL_API_KEY=<your-api-key>
RERANKER_MODEL_BASE_URL=<your-reranker-api-url>
RERANKER_MODEL_ID=Qwen/Qwen3-Reranker
RERANKER_MODEL_API_KEY=<your-api-key>
```

The published benchmark numbers used OpenAI-compatible endpoints with the model IDs above.

## Corpus Setup

Clone the benchmark repositories and pin them to the expected commits:

```bash
bash eval/setup-corpus.sh
cargo run --release --bin vera-eval -- verify-corpus
```

## Build

```bash
cargo build --release
```

## Run The Full Benchmark

```bash
set -a
source secrets.env
set +a
python3 benchmarks/scripts/run_final_benchmarks.py
```

This command:

1. indexes the benchmark repositories
2. runs the Vera benchmark modes
3. compares them with the stored baselines
4. writes results under `benchmarks/results/`

## Useful Partial Runs

```bash
python3 benchmarks/scripts/run_vera_benchmarks.py --modes bm25-only hybrid-norerank hybrid --skip-index --runs 2
python3 benchmarks/scripts/run_baselines.py --tool all --runs 3
python3 benchmarks/scripts/run_baselines.py --tool ripgrep --runs 3
```

## Local Binary Tuning Loop

Use the local-binary harness when you are comparing candidate builds, older binaries, or retrieval tweaks on the exact same corpus:

```bash
python3 benchmarks/scripts/run_local_binary_benchmarks.py \
  --binary target/release/vera \
  --label my-change \
  --extra-arg=--onnx-jina-cuda
```

If you are comparing quantized and fp16 ONNX runs on CUDA, force the model files explicitly. Do not rely on whichever local model path happens to be active:

```bash
VERA_LOCAL_EMBEDDING_ONNX_FILE=onnx/model_quantized.onnx \
VERA_LOCAL_EMBEDDING_ONNX_DATA_FILE=onnx/model_quantized.onnx_data \
python3 benchmarks/scripts/run_local_binary_benchmarks.py \
  --binary target/release/vera \
  --label quantized-cuda \
  --extra-arg=--onnx-jina-cuda

VERA_LOCAL_EMBEDDING_ONNX_FILE=onnx/model_fp16.onnx \
VERA_LOCAL_EMBEDDING_ONNX_DATA_FILE=onnx/model_fp16.onnx_data \
python3 benchmarks/scripts/run_local_binary_benchmarks.py \
  --binary target/release/vera \
  --label fp16-cuda \
  --extra-arg=--onnx-jina-cuda
```

When you are tuning local GPU batching, keep the output log. A healthy run should not repeatedly print `retrying with smaller batches` on the same repo. Occasional retries are a safety net; the goal is to make the steady-state planner avoid the pathological shape in the first place.

If you are measuring cold indexing performance, clear the persisted scaler state first:

```bash
rm -f ~/.vera/adaptive-batch-scaler.json
```

If you keep the file, that is a warm-start benchmark. That can be useful, but compare warm runs to warm runs and cold runs to cold runs.

Treat local tuning numbers as evidence, not a target to game:

- run the full suite after any query-aware retrieval change
- check per-task diffs, not just the aggregate line
- prefer fixes that generalize to the query class over changes that only move one benchmark task
- if a model path or backend change blocks a rerun, document it instead of folding the gap into the result summary

## Expected Ranges

Results vary with model latency and hardware, but on comparable hardware you should expect roughly:

| Metric | Expected range |
|--------|----------------|
| Vera hybrid Recall@10 | `0.70 - 0.85` |
| Vera hybrid MRR@10 | `0.50 - 0.70` |
| BM25 p95 latency | `1 - 15 ms` |
| Hybrid p95 latency | `3000 - 10000 ms` |
| Index size ratio | `1.0x - 2.0x` |

## Notes

- The published public summary uses the 17-task subset across `ripgrep`, `flask`, and `fastify`.
- Hybrid latency depends heavily on remote model round trips.
- If you hit rate limits, rerun after a cooldown period or benchmark repositories one at a time.
