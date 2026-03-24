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
